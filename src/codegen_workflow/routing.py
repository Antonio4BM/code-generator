"""Deterministic routing helpers for the code-generation graph.

Agent nodes return state updates only. They never select their own
successor. Conditional edges in the graph call these pure functions.

Routing functions must not mutate state.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, cast

from codegen_workflow.state import WorkflowState

# Hard upper bound on coder iterations in the coder-review loop.
MAX_ITERATIONS = 4

# Approved inclusive range for ``max_iterations`` overrides.
MIN_MAX_ITERATIONS = 1
MAX_MAX_ITERATIONS = 10

# Explicit terminal / lifecycle statuses used by the workflow.
STATUS_INVALID_INPUT = "invalid_input"
STATUS_PLANNER_FAILED = "planner_failed"
STATUS_CODER_FAILED = "coder_failed"
STATUS_VERIFICATION_FAILED = "verification_failed"
STATUS_REVIEWER_FAILED = "reviewer_failed"
STATUS_ABORTED = "aborted"
STATUS_MAX_ITERATIONS = "max_iterations_reached"
STATUS_PACKAGING_FAILED = "packaging_failed"
STATUS_COMPLETED = "completed"

# Named graph targets used by conditional edges.
InitNext = Literal["planner", "__end__"]
CoderGateNext = Literal["reviewer", "planner", "__end__"]
ReviewerGateNext = Literal["package_project", "planner", "__end__"]
PlannerNext = Literal["coder", "__end__"]


def validate_max_iterations(value: int) -> int:
    """Validate that ``max_iterations`` is within the approved range.

    Args:
        value: Proposed maximum implementation-attempt count.

    Returns:
        The validated positive integer.

    Raises:
        ValueError: If ``value`` is outside
            ``[MIN_MAX_ITERATIONS, MAX_MAX_ITERATIONS]``.
    """
    if value < MIN_MAX_ITERATIONS or value > MAX_MAX_ITERATIONS:
        raise ValueError(
            f"max_iterations must be between {MIN_MAX_ITERATIONS} and {MAX_MAX_ITERATIONS}, got {value}"
        )
    return value


def _decision_value(raw: Mapping[str, object] | None) -> str:
    """Extract a normalized decision string from a human decision dict.

    Args:
        raw: Decision payload from a human gate, if present.

    Returns:
        Lower-cased decision string, or empty string when missing.
    """
    if not raw:
        return ""
    value = raw.get("decision", "")
    if isinstance(value, str):
        return value.strip().lower()
    return ""


def iteration_limit_reached(state: Mapping[str, Any]) -> bool:
    """Return whether the workflow has exhausted its iteration budget.

    Args:
        state: Current workflow state.

    Returns:
        True when ``iteration`` is greater than or equal to
        ``max_iterations`` (defaulting to :data:`MAX_ITERATIONS`).
    """
    iteration = int(state.get("iteration") or 0)
    max_iterations = int(state.get("max_iterations") or MAX_ITERATIONS)
    return iteration >= max_iterations


def route_after_initialize(state: WorkflowState) -> InitNext:
    """Route after workspace initialization.

    Args:
        state: Current workflow state after ``initialize_workspace``.

    Returns:
        ``\"planner\"`` on success, otherwise ``\"__end__\"`` for
        ``invalid_input`` and other initialization failures.
    """
    status = str(state.get("status") or "")
    if status == STATUS_INVALID_INPUT:
        return "__end__"
    if not state.get("workspace_path") or not state.get("workflow_id"):
        return "__end__"
    return "planner"


def route_after_planner(state: WorkflowState) -> PlannerNext:
    """Route after the planner node completes.

    Normal routing is planner → coder. A planner failure with recorded
    errors routes to a terminal end state.

    Args:
        state: Current workflow state after the planner node.

    Returns:
        ``\"coder\"`` on success, otherwise ``\"__end__\"``.
    """
    planner_errors = state.get("planner_errors") or []
    status = str(state.get("status") or "")
    if planner_errors or status in {
        STATUS_PLANNER_FAILED,
        "failed",
        "error",
        STATUS_INVALID_INPUT,
    }:
        return "__end__"
    if not state.get("plan"):
        return "__end__"
    return "coder"


def route_after_coder_gate(state: WorkflowState) -> CoderGateNext:
    """Route after the coder human-approval gate.

    Decisions:

    * ``approve`` → reviewer
    * ``request_changes`` → planner (revision mode, unless iteration limit)
    * ``replan`` → planner (revision mode, unless iteration limit)
    * ``abort`` → END

    When a loop-back decision would continue past ``max_iterations``,
    routing ends with ``max_iterations_reached`` (status set by the gate).

    Args:
        state: Current workflow state after the coder human gate.

    Returns:
        Next node name or ``\"__end__\"``.
    """
    decision = _decision_value(state.get("coder_human_decision"))

    if decision == "abort":
        return "__end__"
    if decision == "approve":
        return "reviewer"

    if decision in {"request_changes", "replan"} and iteration_limit_reached(state):
        return "__end__"

    if decision in {"request_changes", "replan"}:
        return "planner"

    # Unknown or missing decisions are treated as terminal abort.
    return "__end__"


def route_after_reviewer_gate(state: WorkflowState) -> ReviewerGateNext:
    """Route after the reviewer human-approval gate.

    Decisions:

    * ``approve`` → package_project
    * ``request_changes`` → planner (revision mode, unless iteration limit)
    * ``replan`` → planner (revision mode, unless iteration limit)
    * ``abort`` → END

    Args:
        state: Current workflow state after the reviewer human gate.

    Returns:
        Next node name or ``\"__end__\"``.
    """
    decision = _decision_value(state.get("reviewer_human_decision"))

    if decision == "abort":
        return "__end__"
    if decision == "approve":
        return "package_project"

    if decision in {"request_changes", "replan"} and iteration_limit_reached(state):
        return "__end__"

    if decision in {"request_changes", "replan"}:
        return "planner"

    return "__end__"


def status_for_terminal_gate(
    gate: Literal["coder", "reviewer"],
    state: Mapping[str, object],
) -> str:
    """Derive the status string after a human-gate decision.

    Args:
        gate: Which human gate produced the decision.
        state: Current workflow state.

    Returns:
        Status label such as ``aborted``, ``max_iterations_reached``, or
        ``awaiting_review``.
    """
    decision_key = (
        "coder_human_decision" if gate == "coder" else "reviewer_human_decision"
    )
    decision_raw = state.get(decision_key)
    decision = _decision_value(
        cast(Mapping[str, object], decision_raw)
        if isinstance(decision_raw, Mapping)
        else None
    )
    if decision == "abort":
        return STATUS_ABORTED
    if decision in {"request_changes", "replan"} and iteration_limit_reached(state):
        return STATUS_MAX_ITERATIONS
    if decision in {"request_changes", "replan"}:
        return "planning"
    if gate == "coder" and decision == "approve":
        return "awaiting_review"
    if gate == "reviewer" and decision == "approve":
        return "packaging"
    status = state.get("status")
    if isinstance(status, str) and status:
        return status
    return "unknown"
