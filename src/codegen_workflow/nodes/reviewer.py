"""Reviewer agent node for read-only evaluation of generated projects.

Evaluates candidate files against the user request, plan, acceptance
criteria, and verification report. Never modifies generated files and
never selects the next graph node.
"""

from __future__ import annotations

from typing import Any

from codegen_workflow.state import WorkflowState


def reviewer_node(state: WorkflowState) -> dict[str, Any]:
    """Review the generated project and return a structured verdict.

    Args:
        state: Current workflow state after coder human approval.

    Returns:
        State update with ``review_report`` and
        ``status=\"awaiting_reviewer_approval\"``.

    Raises:
        NotImplementedError: Until an LLM-backed reviewer is configured.
            Tests should mock this node instead of calling it live.
    """
    if not state.get("workspace_path"):
        raise ValueError("workspace_path is required for review")

    raise NotImplementedError(
        "reviewer_node requires an LLM-backed implementation; mock it in tests"
    )
