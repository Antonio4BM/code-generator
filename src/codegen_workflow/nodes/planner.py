"""Planner agent node for greenfield project planning.

Converts a natural-language software request into a validated
``ProjectPlan``. The node returns a LangGraph state update and never
writes project files or selects the next graph edge.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Protocol

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from codegen_workflow.schemas.plan import (
    PlanValidationError,
    ProjectPlan,
    validate_plan,
)
from codegen_workflow.state import WorkflowState

logger = logging.getLogger(__name__)

# Reject extremely long requirements that would blow the context window.
MAX_USER_REQUEST_CHARS = 20_000

# Bounded retries for structured-output / schema failures only.
MAX_STRUCTURED_OUTPUT_ATTEMPTS = 3

DEFAULT_PLANNER_MODEL = os.environ.get("PLANNER_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = """\
You are the planner for a greenfield code-generation workflow.
There is no existing repository. Design a complete implementation plan
from the user requirement. Do NOT generate source-code file contents.

Follow this process:
1. Analyze the user request.
2. Identify explicit functional requirements.
3. Identify explicit nonfunctional requirements.
4. Record necessary assumptions.
5. Select the simplest justified technology stack.
6. Define the project architecture.
7. Divide the requirement into epics.
8. Divide epics into user stories.
9. Divide stories into implementation tasks.
10. Define every file required by the project.
11. Define task and file dependencies.
12. Define measurable acceptance criteria.
13. Define installation, execution, and validation commands.

Rules:
- Prefer the simplest stack that satisfies the requirement.
- Do not introduce microservices, message brokers, Kubernetes,
  distributed storage, or other complex infrastructure unless the
  user requirement explicitly justifies them.
- Every path in file_manifest must be relative to the project root
  and must not contain '..'.
- project_name must be filesystem-safe (letter first; letters, digits,
  underscores, hyphens only).
- Every story must reference an existing epic id.
- Every task must reference an existing story id.
- Task dependencies must reference existing task ids and must be acyclic.
- Every story that has a source task must also have a test task.
- Acceptance criteria must be measurable (not vague phrases like
  'works well' or 'looks good').
- file_manifest must include project documentation (e.g. README.md),
  dependency configuration (e.g. requirements.txt or pyproject.toml),
  and automated tests.
- validation_commands must be non-empty.
- Use task_type values: configuration, source, test, documentation,
  container, validation.
"""


class StructuredPlannerModel(Protocol):
    """Minimal protocol for a structured-output planner model."""

    def invoke(self, input: Any, config: Any | None = None, **kwargs: Any) -> Any:
        """Invoke the model and return a ProjectPlan or raw payload."""


def _error(
    error_type: str,
    message: str,
    *,
    details: list[str] | None = None,
) -> dict[str, Any]:
    """Build a typed planner failure state update."""
    record: dict[str, Any] = {"type": error_type, "message": message}
    if details:
        record["details"] = details
    return {
        "plan": {},
        "planner_errors": [record],
        "status": "planner_failed",
    }


def validate_user_request(user_request: str | None) -> str | dict[str, Any]:
    """Validate planner input and return the cleaned request or an error update.

    Args:
        user_request: Raw user requirement from workflow state.

    Returns:
        Stripped request string on success, otherwise a planner failure update.
    """
    if user_request is None:
        return _error(
            "invalid_input",
            "user_request is required and must be non-empty",
        )
    cleaned = user_request.strip()
    if not cleaned:
        return _error(
            "invalid_input",
            "user_request is required and must be non-empty",
        )
    if len(cleaned) > MAX_USER_REQUEST_CHARS:
        return _error(
            "invalid_input",
            f"user_request exceeds the {MAX_USER_REQUEST_CHARS} character limit",
        )
    return cleaned


def build_planner_prompt(
    user_request: str,
    planner_feedback: list[str] | None = None,
) -> str:
    """Build the human prompt for the planner model.

    Args:
        user_request: Validated plain-text software requirement.
        planner_feedback: Optional feedback from human/reviewer replanning.

    Returns:
        Prompt text that includes the request and any feedback.
    """
    sections = [
        "## User requirement",
        user_request.strip(),
    ]
    feedback = [
        item.strip() for item in (planner_feedback or []) if item and item.strip()
    ]
    if feedback:
        sections.append("## Planner feedback from prior review (must address)")
        for index, item in enumerate(feedback, start=1):
            sections.append(f"{index}. {item}")
    else:
        sections.append("## Planner feedback from prior review")
        sections.append("None.")
    sections.append("Return a complete ProjectPlan that satisfies every constraint.")
    return "\n\n".join(sections)


def create_planner_llm(
    *,
    model_name: str | None = None,
    temperature: float = 0.0,
) -> Any:
    """Create a ChatOpenAI model bound to the ProjectPlan schema.

    Args:
        model_name: Optional OpenAI-compatible model id.
        temperature: Sampling temperature (prefer 0 for planning).

    Returns:
        A runnable that returns a ``ProjectPlan`` instance.
    """
    from langchain_openai import ChatOpenAI

    name = model_name or DEFAULT_PLANNER_MODEL
    base: BaseChatModel = ChatOpenAI(model=name, temperature=temperature)
    return base.with_structured_output(ProjectPlan)


def _coerce_plan(raw: Any) -> ProjectPlan:
    """Coerce model output into a ProjectPlan instance.

    Args:
        raw: Structured output from the model.

    Returns:
        Parsed ``ProjectPlan``.

    Raises:
        ValidationError: If the payload cannot be parsed as ProjectPlan.
        TypeError: If the payload type is unsupported.
    """
    if isinstance(raw, ProjectPlan):
        return raw
    if isinstance(raw, dict):
        return ProjectPlan.model_validate(raw)
    raise TypeError(f"unsupported structured output type: {type(raw)!r}")


def invoke_planner_model(
    llm: StructuredPlannerModel,
    user_request: str,
    planner_feedback: list[str] | None = None,
    *,
    max_attempts: int = MAX_STRUCTURED_OUTPUT_ATTEMPTS,
) -> ProjectPlan:
    """Invoke the planner model with bounded retries and validate the plan.

    Args:
        llm: Structured-output model returning ProjectPlan-shaped data.
        user_request: Validated user requirement.
        planner_feedback: Optional replanning feedback.
        max_attempts: Maximum structured-output attempts.

    Returns:
        A deterministically validated ``ProjectPlan``.

    Raises:
        TimeoutError: When the model call times out.
        ValidationError: When structured output fails schema parsing.
        PlanValidationError: When the plan fails domain validation.
        TypeError: When the model returns an unexpected type.
        Exception: Propagates unexpected model errors after retries are exhausted
            for structured-output style failures only.
    """
    prompt = build_planner_prompt(user_request, planner_feedback)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]

    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            raw = llm.invoke(messages)
            plan = _coerce_plan(raw)
            return validate_plan(plan)
        except TimeoutError:
            raise
        except (ValidationError, PlanValidationError, TypeError) as exc:
            last_error = exc
            logger.warning(
                "planner structured output attempt %s/%s failed: %s",
                attempt,
                max_attempts,
                exc,
            )
            # Ask the model to correct the previous failure on retry.
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=prompt),
                HumanMessage(
                    content=(
                        "Previous plan was invalid. Fix every issue and "
                        f"return a complete ProjectPlan.\nErrors:\n{exc}"
                    )
                ),
            ]
            continue
    assert last_error is not None
    raise last_error


def planner_node(
    state: WorkflowState,
    *,
    llm: StructuredPlannerModel | None = None,
) -> dict[str, Any]:
    """Produce a validated project plan from the user request.

    Reads ``user_request`` and optional ``planner_feedback`` from state.
    Returns a state update and does not select the next graph node or
    write files into the workspace.

    Args:
        state: Current workflow state.
        llm: Optional injectable structured model (for tests). When
            omitted, a default ChatOpenAI structured-output chain is used.

    Returns:
        State update with ``plan``, ``planner_errors``, and ``status``.
        On success ``status`` is ``\"coding\"``. On failure ``status`` is
        ``\"planner_failed\"`` with typed ``planner_errors``.
    """
    validated = validate_user_request(state.get("user_request"))
    if isinstance(validated, dict):
        return validated

    feedback = list(state.get("planner_feedback") or [])

    try:
        model = llm if llm is not None else create_planner_llm()
        plan = invoke_planner_model(model, validated, feedback)
    except TimeoutError as exc:
        logger.exception("planner model timed out")
        return _error("model_timeout", f"planner model timed out: {exc}")
    except PlanValidationError as exc:
        return _error(
            "validation_error",
            "plan failed deterministic validation",
            details=list(exc.errors),
        )
    except ValidationError as exc:
        return _error(
            "invalid_structured_output",
            "model returned data that does not match ProjectPlan",
            details=[str(exc)],
        )
    except TypeError as exc:
        return _error(
            "invalid_structured_output",
            f"model returned unsupported output type: {exc}",
        )
    except Exception as exc:  # noqa: BLE001 - map unknown failures for routing
        message = str(exc).lower()
        if "timeout" in message:
            return _error("model_timeout", f"planner model timed out: {exc}")
        if any(
            token in message
            for token in ("impossible", "unsupported", "cannot satisfy", "infeasible")
        ):
            return _error(
                "unsupported_requirement",
                f"requirement appears unsupported or impossible: {exc}",
            )
        logger.exception("planner failed with unexpected error")
        return _error("planner_error", f"planner failed: {exc}")

    return {
        "plan": plan.model_dump(),
        "planner_errors": [],
        "status": "coding",
    }
