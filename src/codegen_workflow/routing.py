"""Deterministic routing helpers for the code-generation graph.

Agent nodes return state updates only. They never select their own
successor. Conditional edges in the graph call these pure functions.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from codegen_workflow.state import WorkflowState

# Hard upper bound on coder iterations in the coder-review loop.
MAX_ITERATIONS = 4

# Named graph targets used by conditional edges.
CoderGateNext = Literal["reviewer", "coder", "planner", "__end__"]
ReviewerGateNext = Literal["package_project", "coder", "planner", "__end__"]
PlannerNext = Literal["coder", "__end__"]


def _decision_value(raw: dict[str, Any] | None) -> str:
    """Extract a normalized decision string from a human decision dict.

    Args:
        raw: Decision payload from a human gate, if present.

    Returns:
        Lower-cased decision string, or empty string when missing.
    """
    if not raw:
        return ""
    return str(raw.get("decision", "")).strip().lower()


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
    if planner_errors or status in {"planner_failed", "failed", "error"}:
        return "__end__"
    if not state.get("plan"):
        return "__end__"
    return "coder"


def route_after_coder_gate(state: WorkflowState) -> CoderGateNext:
    """Route after the coder human-approval gate.

    Decisions:

    * ``approve`` → reviewer
    * ``request_changes`` → coder (unless iteration limit reached)
    * ``replan`` → planner (unless iteration limit reached)
    * ``abort`` → END

    When a loop-back decision would continue past ``max_iterations``,
    routing escalates to END so generated artifacts remain inspectable.

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

    if decision == "request_changes":
        return "coder"
    if decision == "replan":
        return "planner"

    # Unknown or missing decisions are treated as terminal abort.
    return "__end__"


def route_after_reviewer_gate(state: WorkflowState) -> ReviewerGateNext:
    """Route after the reviewer human-approval gate.

    Decisions:

    * ``approve`` → package_project
    * ``request_changes`` → coder (unless iteration limit reached)
    * ``replan`` → planner (unless iteration limit reached)
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

    if decision == "request_changes":
        return "coder"
    if decision == "replan":
        return "planner"

    return "__end__"


def status_for_terminal_gate(
    gate: Literal["coder", "reviewer"],
    state: Mapping[str, Any],
) -> str:
    """Derive a terminal status string after a human-gate decision.

    Args:
        gate: Which human gate produced the decision.
        state: Current workflow state.

    Returns:
        Status label such as ``aborted``, ``escalated``, or
        ``awaiting_review``.
    """
    decision_key = (
        "coder_human_decision" if gate == "coder" else "reviewer_human_decision"
    )
    decision = _decision_value(state.get(decision_key))
    if decision == "abort":
        return "aborted"
    if decision in {"request_changes", "replan"} and iteration_limit_reached(state):
        return "escalated"
    if gate == "coder" and decision == "approve":
        return "awaiting_review"
    if gate == "reviewer" and decision == "approve":
        return "packaging"
    return str(state.get("status") or "unknown")
