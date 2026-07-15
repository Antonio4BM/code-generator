"""Reviewer agent node for read-only evaluation of generated projects.

Inspects candidate files against the user request, approved plan,
acceptance criteria, verification report, previous findings, and coder
resolution claims. Never modifies generated files and never selects the
next graph edge.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Protocol

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import ValidationError

from codegen_workflow.schemas.plan import ProjectPlan
from codegen_workflow.schemas.review import ReviewFinding, ReviewReport, ReviewVerdict
from codegen_workflow.state import WorkflowState
from codegen_workflow.tools.readonly import ReadOnlyWorkspaceTools
from codegen_workflow.tools.workspace import WorkspaceLimitError, WorkspaceSecurityError
from codegen_workflow.workspace import candidate_dir

logger = logging.getLogger(__name__)

DEFAULT_REVIEWER_MODEL = os.environ.get("REVIEWER_MODEL", "gpt-4o-mini")
MAX_TOOL_CALLS = 40
MAX_MODEL_RETRIES = 3

SYSTEM_PROMPT = """\
You are an independent, read-only reviewer for a greenfield
code-generation workflow. You may inspect files with the provided tools
but you MUST NOT modify files.

Evaluate the candidate project against:
1. The original user request.
2. The approved project plan.
3. Every planner acceptance criterion (report each as true/false).
4. The generated file manifest.
5. Automated verification results.
6. Previous review findings.
7. Coder feedback-resolution claims (treat unverified claims as defects).

Review process:
1. Read the plan and verification report.
2. Inspect every required manifest file.
3. Evaluate every acceptance criterion separately.
4. Verify claimed feedback resolutions against the code.
5. Identify correctness defects.
6. Identify security risks (hardcoded secrets, unsafe defaults, etc.).
7. Evaluate test quality, not only quantity.
8. Detect missing or incomplete documentation.
9. Produce a structured ReviewReport.

Verdict rules:
- approve: only when there are no blocking or major findings, automated
  checks pass, every mandatory acceptance criterion passes, and every
  mandatory manifest file exists.
- request_changes: plan remains valid but implementation must change.
- replan: approved architecture/task decomposition is insufficient or
  contradictory.

Findings must be concrete, with evidence, file/line when available, and
an actionable recommendation. Never emit generic advice such as
"Improve code quality."
"""


class ReviewerModel(Protocol):
    """Minimal protocol for tool-calling + structured-output models."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        """Bind read-only tools."""

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        """Bind a structured output schema."""

    def invoke(self, input: Any, config: Any | None = None, **kwargs: Any) -> Any:
        """Invoke the model."""


def create_reviewer_llm(
    *,
    model_name: str | None = None,
    temperature: float = 0.0,
) -> BaseChatModel:
    """Create the default ChatOpenAI model used by the reviewer."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model_name or DEFAULT_REVIEWER_MODEL,
        temperature=temperature,
    )


def collect_acceptance_criteria(plan: ProjectPlan) -> dict[str, str]:
    """Build stable acceptance-criterion ids mapped to criterion text.

    Args:
        plan: Validated project plan.

    Returns:
        Ordered mapping such as ``epic:E1:0`` → criterion text.
    """
    criteria: dict[str, str] = {}
    for epic in plan.epics:
        for index, text in enumerate(epic.acceptance_criteria):
            criteria[f"epic:{epic.id}:{index}"] = text
    for story in plan.stories:
        for index, text in enumerate(story.acceptance_criteria):
            criteria[f"story:{story.id}:{index}"] = text
    for task in plan.tasks:
        for index, text in enumerate(task.acceptance_criteria):
            criteria[f"task:{task.id}:{index}"] = text
    return criteria


def check_manifest_results(
    tools: ReadOnlyWorkspaceTools,
    plan: ProjectPlan,
) -> dict[str, bool]:
    """Return whether each planned manifest path exists on disk."""
    existing = set(tools.list_files())
    return {
        spec.path.replace("\\", "/"): spec.path.replace("\\", "/") in existing
        for spec in plan.file_manifest
    }


def verification_passed(verification_report: dict[str, Any] | None) -> bool:
    """Return whether the automated verification report indicates success."""
    if not verification_report:
        return False
    if verification_report.get("passed") is True:
        return True
    return str(verification_report.get("overall_status") or "").lower() == "passed"


def build_reviewer_prompt(
    *,
    user_request: str,
    plan: ProjectPlan,
    generated_files: list[str],
    verification_report: dict[str, Any],
    coder_result: dict[str, Any],
    previous_review_report: dict[str, Any] | None,
    acceptance_criteria: dict[str, str],
    manifest_results: dict[str, bool],
) -> str:
    """Build the human prompt for the reviewer model."""
    previous = previous_review_report or {}
    return "\n\n".join(
        [
            f"## User request\n{user_request.strip()}",
            "## Plan\n" + plan.model_dump_json(indent=2),
            "## Generated files (state)\n"
            + ("\n".join(f"- {path}" for path in generated_files) or "None."),
            "## Deterministic manifest existence\n"
            + json.dumps(manifest_results, indent=2, sort_keys=True),
            "## Acceptance criteria to evaluate\n"
            + json.dumps(acceptance_criteria, indent=2, sort_keys=True),
            "## Verification report\n"
            + json.dumps(verification_report or {}, indent=2, sort_keys=True),
            "## Coder result (including feedback_resolutions)\n"
            + json.dumps(coder_result or {}, indent=2, sort_keys=True),
            "## Previous review report\n"
            + json.dumps(previous, indent=2, sort_keys=True),
            "Use read-only tools to inspect the candidate files, then produce "
            "a complete ReviewReport. Every acceptance criterion id listed "
            "above must appear in acceptance_criteria_results.",
        ]
    )


def run_readonly_tool_loop(
    llm: Any,
    tools: ReadOnlyWorkspaceTools,
    messages: list[Any],
    *,
    max_tool_calls: int = MAX_TOOL_CALLS,
    max_model_retries: int = MAX_MODEL_RETRIES,
) -> list[Any]:
    """Execute a bounded read-only model–tool inspection loop.

    Args:
        llm: Chat model supporting ``bind_tools``.
        tools: Read-only workspace toolkit.
        messages: Seeded conversation messages.
        max_tool_calls: Hard cap on tool invocations.
        max_model_retries: Retries for transient model failures.

    Returns:
        Updated message list after the model stops calling tools.
    """
    bound = llm.bind_tools(tools.as_langchain_tools())
    tool_call_count = 0

    while True:
        response: Any | None = None
        last_error: Exception | None = None
        for _attempt in range(1, max_model_retries + 1):
            try:
                response = bound.invoke(messages)
                break
            except TimeoutError:
                raise
            except Exception as exc:  # noqa: BLE001 - bounded retry wrapper
                last_error = exc
                logger.warning("reviewer tool-loop invoke failed: %s", exc)
                continue
        if response is None:
            assert last_error is not None
            raise last_error

        messages.append(response)
        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            break

        for call in tool_calls:
            if tool_call_count >= max_tool_calls:
                messages.append(
                    HumanMessage(
                        content=(
                            "Tool-call budget exhausted. Stop calling tools and "
                            "continue with the ReviewReport based on evidence so far."
                        )
                    )
                )
                return messages
            tool_call_count += 1
            name = (
                call.get("name")
                if isinstance(call, dict)
                else getattr(call, "name", "")
            )
            call_id = (
                call.get("id") if isinstance(call, dict) else getattr(call, "id", "")
            ) or f"call_{tool_call_count}"
            args = (
                call.get("args")
                if isinstance(call, dict)
                else getattr(call, "args", {})
            ) or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            try:
                result_text = tools.invoke_tool(str(name), dict(args))
            except (
                WorkspaceSecurityError,
                WorkspaceLimitError,
                FileNotFoundError,
                ValueError,
                OSError,
            ) as exc:
                result_text = f"ERROR: {exc}"
            messages.append(ToolMessage(content=result_text, tool_call_id=str(call_id)))

    return messages


def _coerce_report(raw: Any) -> ReviewReport:
    """Coerce model output into a ReviewReport instance."""
    if isinstance(raw, ReviewReport):
        return raw
    if isinstance(raw, dict):
        return ReviewReport.model_validate(raw)
    raise TypeError(f"unsupported review output type: {type(raw)!r}")


def enforce_review_invariants(
    report: ReviewReport,
    *,
    acceptance_criteria: dict[str, str],
    manifest_results: dict[str, bool],
    verification_ok: bool,
    reviewed_files: list[str],
) -> ReviewReport:
    """Apply deterministic approve gates and fill missing criterion keys.

    The model may propose a verdict, but ``approve`` is allowed only when
    automated gates pass and no blocking/major findings remain.
    """
    # Ensure every criterion key has an explicit boolean result.
    ac_results = dict(report.acceptance_criteria_results)
    for key in acceptance_criteria:
        if key not in ac_results:
            ac_results[key] = False

    merged_manifest = dict(manifest_results)
    merged_manifest.update(report.manifest_results or {})
    # Deterministic existence wins for required paths.
    for path, exists in manifest_results.items():
        merged_manifest[path] = exists

    findings = list(report.findings)
    finding_ids = {finding.finding_id for finding in findings}

    if not verification_ok and "VERIFICATION_FAILED" not in finding_ids:
        findings.append(
            ReviewFinding(
                finding_id="VERIFICATION_FAILED",
                severity="blocking",
                category="testing",
                file=None,
                line=None,
                description="Automated verification did not pass for this candidate.",
                evidence=f"verification_passed={verification_ok}",
                recommendation=(
                    "Fix the failing verification commands and re-run until they pass."
                ),
            )
        )

    missing_paths = [path for path, exists in merged_manifest.items() if not exists]
    if missing_paths and "MISSING_MANIFEST_FILES" not in finding_ids:
        findings.append(
            ReviewFinding(
                finding_id="MISSING_MANIFEST_FILES",
                severity="blocking",
                category="requirements",
                file=missing_paths[0],
                line=None,
                description="One or more mandatory manifest files are missing.",
                evidence="missing: " + ", ".join(missing_paths),
                recommendation="Create every path listed in the planner file_manifest.",
            )
        )

    has_blocking = any(finding.severity == "blocking" for finding in findings)
    has_major = any(finding.severity == "major" for finding in findings)
    all_criteria_pass = bool(ac_results) and all(ac_results.values())
    all_manifest_present = all(merged_manifest.values()) if merged_manifest else False

    verdict: ReviewVerdict = report.verdict
    if verdict == "approve":
        if (
            has_blocking
            or has_major
            or not verification_ok
            or not all_criteria_pass
            or not all_manifest_present
        ):
            verdict = "request_changes"
    elif verdict not in {"request_changes", "replan"}:
        verdict = "request_changes"

    files = list(dict.fromkeys([*reviewed_files, *report.reviewed_files]))

    return ReviewReport(
        verdict=verdict,
        acceptance_criteria_results=ac_results,
        manifest_results=merged_manifest,
        reviewed_files=files,
        findings=findings,
        residual_risks=list(report.residual_risks),
        summary=report.summary,
    )


def reviewer_node(
    state: WorkflowState,
    *,
    llm: ReviewerModel | None = None,
    max_tool_calls: int = MAX_TOOL_CALLS,
    max_model_retries: int = MAX_MODEL_RETRIES,
) -> dict[str, Any]:
    """Review the generated project and return a structured verdict.

    Args:
        state: Current workflow state after coder human approval.
        llm: Optional injectable model (for tests).
        max_tool_calls: Bound on read-only tool calls.
        max_model_retries: Bound on model invoke retries.

    Returns:
        State update with ``review_report`` and
        ``status=\"awaiting_reviewer_approval\"``.

    Raises:
        ValueError: If ``workspace_path`` is missing.
    """
    workspace_path = state.get("workspace_path")
    if not workspace_path:
        raise ValueError("workspace_path is required for review")

    plan_raw = state.get("plan") or {}
    try:
        plan = ProjectPlan.model_validate(plan_raw)
    except ValidationError as exc:
        report = ReviewReport(
            verdict="replan",
            acceptance_criteria_results={},
            manifest_results={},
            reviewed_files=[],
            findings=[
                ReviewFinding(
                    finding_id="INVALID_PLAN",
                    severity="blocking",
                    category="architecture",
                    file=None,
                    line=None,
                    description="The workflow plan is invalid or incomplete for review.",
                    evidence=str(exc),
                    recommendation="Return to planning and produce a valid ProjectPlan.",
                )
            ],
            residual_risks=["Plan validation failed before code review."],
            summary="Replan required due to invalid plan payload.",
        )
        return {
            "review_report": report.model_dump(),
            "status": "awaiting_reviewer_approval",
        }

    verification_report = dict(state.get("verification_report") or {})
    coder_result = dict(state.get("coder_result") or {})
    previous_raw = state.get("previous_review_report")
    if previous_raw is None:
        previous_raw = state.get("review_report") or {}
    previous_review_report: dict[str, Any] | None
    if isinstance(previous_raw, dict):
        previous_review_report = dict(previous_raw)
    else:
        previous_review_report = {}
    generated_files = list(state.get("generated_files") or [])
    user_request = str(state.get("user_request") or plan.objective)

    tools = ReadOnlyWorkspaceTools(
        candidate_dir(workspace_path),
        verification_report=verification_report,
    )
    acceptance_criteria = collect_acceptance_criteria(plan)
    manifest_results = check_manifest_results(tools, plan)
    verification_ok = verification_passed(verification_report)

    messages: list[Any] = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=build_reviewer_prompt(
                user_request=user_request,
                plan=plan,
                generated_files=generated_files,
                verification_report=verification_report,
                coder_result=coder_result,
                previous_review_report=(
                    previous_review_report if previous_review_report else None
                ),
                acceptance_criteria=acceptance_criteria,
                manifest_results=manifest_results,
            )
        ),
    ]

    model: Any = llm if llm is not None else create_reviewer_llm()

    try:
        messages = run_readonly_tool_loop(
            model,
            tools,
            messages,
            max_tool_calls=max_tool_calls,
            max_model_retries=max_model_retries,
        )
        structured = model.with_structured_output(ReviewReport)
        messages.append(
            HumanMessage(
                content=(
                    "Return the final ReviewReport now. Include every acceptance "
                    "criterion id with an explicit boolean result, concrete "
                    "findings with evidence, and the correct verdict."
                )
            )
        )
        raw_report = structured.invoke(messages)
        report = _coerce_report(raw_report)
    except Exception as exc:  # noqa: BLE001 - map failures into a review report
        logger.exception("reviewer model failed")
        report = ReviewReport(
            verdict="request_changes",
            acceptance_criteria_results={key: False for key in acceptance_criteria},
            manifest_results=manifest_results,
            reviewed_files=list(tools.reviewed_files),
            findings=[
                ReviewFinding(
                    finding_id="REVIEWER_ERROR",
                    severity="blocking",
                    category="maintainability",
                    file=None,
                    line=None,
                    description="The reviewer model failed before producing a report.",
                    evidence=str(exc),
                    recommendation="Retry the review after resolving the model failure.",
                )
            ],
            residual_risks=["Reviewer execution failed."],
            summary=f"Reviewer failed: {exc}",
        )

    final = enforce_review_invariants(
        report,
        acceptance_criteria=acceptance_criteria,
        manifest_results=manifest_results,
        verification_ok=verification_ok,
        reviewed_files=list(tools.reviewed_files) or generated_files,
    )

    return {
        "review_report": final.model_dump(),
        "status": "awaiting_reviewer_approval",
    }


__all__ = [
    "AIMessage",
    "MAX_MODEL_RETRIES",
    "MAX_TOOL_CALLS",
    "build_reviewer_prompt",
    "check_manifest_results",
    "collect_acceptance_criteria",
    "create_reviewer_llm",
    "enforce_review_invariants",
    "reviewer_node",
    "run_readonly_tool_loop",
    "verification_passed",
]
