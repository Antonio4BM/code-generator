"""Planner agent node for greenfield project planning.

Converts a natural-language software request into a validated project
plan. This module owns the planner interface consumed by the graph; a
full LLM-backed implementation may replace the default placeholder.
"""

from __future__ import annotations

from typing import Any

from codegen_workflow.state import WorkflowState


def planner_node(state: WorkflowState) -> dict[str, Any]:
    """Produce a validated project plan from the user request.

    Reads ``user_request`` and optional ``planner_feedback`` from state.
    Returns a state update and does not select the next graph node.

    Args:
        state: Current workflow state.

    Returns:
        State update with ``plan``, ``planner_errors``, and ``status``.

    Raises:
        ValueError: If ``user_request`` is missing or empty.
        NotImplementedError: Until an LLM-backed planner is configured.
            Tests should mock this node instead of calling it live.
    """
    user_request = (state.get("user_request") or "").strip()
    if not user_request:
        return {
            "plan": {},
            "planner_errors": [
                {
                    "type": "invalid_input",
                    "message": "user_request is required and must be non-empty",
                }
            ],
            "status": "planner_failed",
        }

    # Intentionally unimplemented for live LLM calls. Graph tests mock
    # this function. Production deployments wire a structured-output model.
    raise NotImplementedError(
        "planner_node requires an LLM-backed implementation; mock it in tests"
    )
