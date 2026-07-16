"""Planner agent node for greenfield project planning.

Converts a natural-language software request into a validated
``ProjectPlan``. Supports initial planning and revision planning when
a human ``change_request`` is present. Returns a LangGraph state update
and never writes project files or selects the next graph edge.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Protocol

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from codegen_workflow.revision import plan_diff_payload
from codegen_workflow.schemas.plan import (
    PlanValidationError,
    ProjectPlan,
    validate_plan,
)
from codegen_workflow.state import WorkflowState
from codegen_workflow.workspace import candidate_dir

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
- Acceptance criteria must be measurable (not vague phrases like
  'works well' or 'looks good').
- file_manifest must include project documentation (e.g. README.md).
- Automated tests, paired source/test tasks, and dependency-configuration
  files (requirements.txt, pyproject.toml, package.json) are OPTIONAL.
  Prefer including them for application/API projects when practical, but
  do not block planning if they are omitted.
- Prefer shell-executable validation_commands such as
  ``python3 -c "from pathlib import Path; assert Path('index.html').is_file()"``
  or ``pytest -q`` rather than browser-open instructions.
- validation_commands must be non-empty shell-executable commands
  (never browser-open instructions like "open index.html in Chrome").
- Use task_type values: configuration, source, test, documentation,
  container, validation.
"""

REVISION_SYSTEM_ADDENDUM = """\
## Revision mode

You are revising an existing generated project.

Produce a complete revised ProjectPlan representing the entire desired
project after applying the requested change.

The existing plan is context, not an immutable constraint.

You may add, remove, rename, or reorganize files, dependencies, tasks,
acceptance criteria, commands, and architectural components when required.

Preserve unaffected parts of the current plan.

Remove obsolete files and requirements from the revised plan.

Do not return an incremental patch. Return the complete authoritative
ProjectPlan after the revision.

## Application-code transitions

When the requested change adds an API, backend, server, database, or
other application code to a previously static markup/HTML site:

- Set language/framework/architecture to the backend stack (for example
  Python + FastAPI, or Node + Express).
- Keep useful static assets (index.html, styles.css) when still needed,
  and add API modules beside them.
- Dependency configuration files and automated tests are OPTIONAL but
  recommended when practical.
- validation_commands MUST be shell-executable (for example
  ``python3 -c "..."`` or ``pytest -q``). Never use browser-open
  instructions.
"""

_APPLICATION_CHANGE_HINTS = (
    "api",
    "backend",
    "server",
    "fastapi",
    "flask",
    "django",
    "express",
    "endpoint",
    "rest",
    "graphql",
    "database",
    "db ",
    "auth",
    "jwt",
    "microservice",
    "application code",
    "python app",
    "node app",
)


def _feedback_implies_application_code(feedback: str) -> bool:
    """Return whether human feedback asks for backend/API application code."""
    text = f" {feedback.lower()} "
    return any(hint in text for hint in _APPLICATION_CHANGE_HINTS)


def _plan_dict_is_static_markup(plan: dict[str, Any] | None) -> bool:
    """Best-effort check whether the current plan is a static markup site."""
    if not plan:
        return False
    try:
        from codegen_workflow.schemas.plan import is_static_markup_project

        return is_static_markup_project(ProjectPlan.model_validate(plan))
    except Exception:  # noqa: BLE001 - heuristic fallback for partial plans
        language = str(plan.get("language") or "").strip().lower()
        architecture = str(plan.get("architecture_pattern") or "").strip().lower()
        return language in {
            "html",
            "css",
            "markdown",
            "static",
            "static html",
            "static-site",
            "static site",
        } or any(
            hint in architecture
            for hint in ("static", "landing page", "markup")
        )


class StructuredPlannerModel(Protocol):
    """Minimal protocol for a structured-output planner model."""

    def invoke(self, input: Any, config: Any | None = None, **kwargs: Any) -> Any:
        """Invoke the model and return a ProjectPlan or raw payload."""


def _error(
    error_type: str,
    message: str,
    *,
    details: list[str] | None = None,
    preserve_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a typed planner failure state update.

    Args:
        error_type: Machine-readable error type.
        message: Human-readable error message.
        details: Optional detail strings.
        preserve_plan: When set (revision failures), keep the current
            authoritative plan instead of clearing it.
    """
    record: dict[str, Any] = {"type": error_type, "message": message}
    if details:
        record["details"] = details
    update: dict[str, Any] = {
        "planner_errors": [record],
        "status": "planner_failed",
    }
    if preserve_plan is not None:
        update["plan"] = preserve_plan
    else:
        update["plan"] = {}
    return update


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


def _file_tree_from_state(state: WorkflowState) -> list[str]:
    """List candidate files from state or the workspace on disk."""
    generated = list(state.get("generated_files") or [])
    if generated:
        return sorted({str(path).replace("\\", "/") for path in generated})
    workspace_path = state.get("workspace_path")
    if not workspace_path:
        return []
    root = candidate_dir(workspace_path)
    if not Path(root).exists():
        return []
    files: list[str] = []
    for path in sorted(Path(root).rglob("*")):
        if path.is_file():
            files.append(path.relative_to(root).as_posix())
    return files


def is_revision_request(state: WorkflowState) -> bool:
    """Return whether the planner should run in revision mode.

    Args:
        state: Current workflow state.

    Returns:
        True when an authoritative plan and change_request are present.
    """
    plan = state.get("plan") or {}
    change_request = state.get("change_request") or {}
    return bool(plan) and bool(change_request)


def build_planner_prompt(
    user_request: str,
    planner_feedback: list[str] | None = None,
) -> str:
    """Build the human prompt for initial planning mode.

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


def build_revision_prompt(state: WorkflowState, user_request: str) -> str:
    """Build the human prompt for planner revision mode.

    Args:
        state: Workflow state containing plan, change_request, and reports.
        user_request: Validated original requirement.

    Returns:
        Prompt that includes the current plan and requested change.
    """
    change_request = state.get("change_request") or {}
    feedback = str(change_request.get("feedback") or "").strip() or "None."
    files = _file_tree_from_state(state)
    history = state.get("feedback_history") or []
    history_lines: list[str] = []
    for entry in history[-8:]:
        if not isinstance(entry, dict):
            continue
        history_lines.append(
            f"- gate={entry.get('gate')} decision={entry.get('decision')} "
            f"iteration={entry.get('iteration')}: "
            f"{str(entry.get('feedback') or '').strip() or '(no text)'}"
        )

    current_plan = state.get("plan") or {}
    static_to_app = _plan_dict_is_static_markup(
        current_plan if isinstance(current_plan, dict) else None
    ) and _feedback_implies_application_code(feedback)

    sections = [
        "## Revision mode",
        (
            "Produce a complete revised ProjectPlan for the entire desired "
            "project after applying the requested change."
        ),
        "## Original user requirement",
        user_request.strip(),
        "## Current ProjectPlan",
        json.dumps(current_plan, indent=2, default=str),
        "## Current generated files",
        ("\n".join(f"- {path}" for path in files) if files else "None."),
        "## Requested change",
        (
            f"source_gate={change_request.get('source_gate')}\n"
            f"iteration={change_request.get('iteration')}\n"
            f"feedback={feedback}"
        ),
        "## Latest verification report",
        json.dumps(state.get("verification_report") or {}, indent=2, default=str),
        "## Latest review report",
        json.dumps(state.get("review_report") or {}, indent=2, default=str),
        "## Recent feedback history",
        ("\n".join(history_lines) if history_lines else "None."),
    ]

    if static_to_app:
        sections.append(
            "## Critical: static site → application/API upgrade\n"
            "The current plan is a static markup site, but the requested "
            "change adds an API or backend.\n\n"
            "Your revised plan SHOULD include:\n"
            "1. Backend language/framework (for example Python + FastAPI).\n"
            "2. API modules next to any retained landing-page assets.\n"
            "3. Shell-executable validation_commands "
            "(for example: python3 -c \"...\" or pytest -q).\n\n"
            "Dependency configuration files and automated tests are "
            "OPTIONAL — include them when practical, but do not fail the "
            "revision solely to invent them.\n"
            "Do not invent browser-open validation commands."
        )

    sections.append(
        "Return a complete ProjectPlan. Remove obsolete files from "
        "file_manifest when a feature is removed. Prefer shell-executable "
        "validation_commands."
    )
    return "\n\n".join(sections)


def create_planner_llm(
    *,
    model_name: str | None = None,
    temperature: float = 0.0,
) -> Any:
    """Create an AzureChatOpenAI model bound to the ProjectPlan schema.

    Args:
        model_name: Optional Azure deployment name.
        temperature: Sampling temperature (prefer 0 for planning).

    Returns:
        A runnable that returns a ``ProjectPlan`` instance.
    """
    from codegen_workflow.llm import create_azure_chat_model

    name = model_name or DEFAULT_PLANNER_MODEL
    base: BaseChatModel = create_azure_chat_model(
        model_name=name,
        temperature=temperature,
    )
    return base.with_structured_output(ProjectPlan)


def _coerce_plan(raw: Any) -> ProjectPlan:
    """Coerce model output into a ProjectPlan instance."""
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
    prompt: str | None = None,
    system_prompt: str = SYSTEM_PROMPT,
    max_attempts: int = MAX_STRUCTURED_OUTPUT_ATTEMPTS,
) -> ProjectPlan:
    """Invoke the planner model with bounded retries and validate the plan.

    Args:
        llm: Structured-output model returning ProjectPlan-shaped data.
        user_request: Validated user requirement.
        planner_feedback: Optional feedback from prior review (initial mode).
        prompt: Optional prebuilt human prompt (revision mode).
        system_prompt: System instructions for the model.
        max_attempts: Maximum structured-output attempts.

    Returns:
        A deterministically validated ``ProjectPlan``.
    """
    human_prompt = prompt or build_planner_prompt(user_request, planner_feedback)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt),
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
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt),
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


def _map_planner_exception(
    exc: Exception,
    *,
    preserve_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Map planner exceptions to typed failure updates."""
    if isinstance(exc, TimeoutError):
        logger.exception("planner model timed out")
        return _error(
            "model_timeout",
            f"planner model timed out: {exc}",
            preserve_plan=preserve_plan,
        )
    if isinstance(exc, PlanValidationError):
        return _error(
            "validation_error",
            "plan failed deterministic validation",
            details=list(exc.errors),
            preserve_plan=preserve_plan,
        )
    if isinstance(exc, ValidationError):
        return _error(
            "invalid_structured_output",
            "model returned data that does not match ProjectPlan",
            details=[str(exc)],
            preserve_plan=preserve_plan,
        )
    if isinstance(exc, TypeError):
        return _error(
            "invalid_structured_output",
            f"model returned unsupported output type: {exc}",
            preserve_plan=preserve_plan,
        )
    message = str(exc).lower()
    if "timeout" in message:
        return _error(
            "model_timeout",
            f"planner model timed out: {exc}",
            preserve_plan=preserve_plan,
        )
    if any(
        token in message
        for token in ("impossible", "unsupported", "cannot satisfy", "infeasible")
    ):
        return _error(
            "unsupported_requirement",
            f"requirement appears unsupported or impossible: {exc}",
            preserve_plan=preserve_plan,
        )
    logger.exception("planner failed with unexpected error")
    return _error(
        "planner_error",
        f"planner failed: {exc}",
        preserve_plan=preserve_plan,
    )


def planner_node(
    state: WorkflowState,
    *,
    llm: StructuredPlannerModel | None = None,
) -> dict[str, Any]:
    """Produce a validated project plan from the user request.

    Runs in initial planning mode or revision mode when ``plan`` and
    ``change_request`` are present. Returns a state update and does not
    select the next graph node or write files into the workspace.

    Args:
        state: Current workflow state.
        llm: Optional injectable structured model (for tests).

    Returns:
        State update with ``plan``, ``planner_errors``, and ``status``.
        Revision successes also set ``previous_plan`` and ``plan_diff``.
    """
    validated = validate_user_request(state.get("user_request"))
    if isinstance(validated, dict):
        return validated

    revision = is_revision_request(state)
    current_plan = dict(state.get("plan") or {}) if revision else None
    feedback = list(state.get("planner_feedback") or [])

    try:
        model = llm if llm is not None else create_planner_llm()
        if revision:
            plan = invoke_planner_model(
                model,
                validated,
                feedback,
                prompt=build_revision_prompt(state, validated),
                system_prompt=SYSTEM_PROMPT + "\n" + REVISION_SYSTEM_ADDENDUM,
            )
        else:
            plan = invoke_planner_model(model, validated, feedback)
    except Exception as exc:  # noqa: BLE001 - map failures for routing
        return _map_planner_exception(exc, preserve_plan=current_plan)

    if revision:
        assert current_plan is not None
        return {
            "previous_plan": current_plan,
            "plan": plan.model_dump(),
            "plan_diff": plan_diff_payload(current_plan, plan),
            "planner_errors": [],
            "status": "coding",
            # Consumed by this revision; next change_request recreates it.
            "change_request": {},
        }

    return {
        "plan": plan.model_dump(),
        "planner_errors": [],
        "previous_plan": {},
        "plan_diff": {},
        "status": "coding",
    }
