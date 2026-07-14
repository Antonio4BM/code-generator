"""Shared typed state for the code-generation workflow.

The graph stores structured artifacts explicitly. A message list is not
used as the primary state representation.
"""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict


class WorkflowState(TypedDict):
    """Typed LangGraph state for one user software request.

    Attributes:
        user_request: Plain-text software requirement from the caller.
        workflow_id: UUID identifying this workflow run.
        workspace_path: Absolute path to the isolated workspace root.
        plan: Validated project plan produced by the planner.
        planner_feedback: Feedback strings used when replanning.
        generated_files: Relative paths of files written by the coder.
        file_hashes: SHA-256 hashes keyed by generated file path.
        coder_result: Structured summary from the latest coder iteration.
        verification_report: Structured report from deterministic checks.
        review_report: Structured report from the reviewer agent.
        feedback_history: Append-only log of human gate decisions.
        coder_human_decision: Latest decision from the coder human gate.
        reviewer_human_decision: Latest decision from the reviewer gate.
        iteration: Number of completed coder iterations.
        max_iterations: Hard upper bound on coder iterations.
        status: Current workflow lifecycle status string.
        artifact_path: Path to the final ZIP archive when packaged.
        artifact_hash: SHA-256 digest of the packaged archive.
        errors: Typed error records accumulated during the run.
        planner_errors: Planner-specific error records, if any.
    """

    user_request: str
    workflow_id: NotRequired[str]
    workspace_path: NotRequired[str]
    plan: NotRequired[dict[str, Any]]
    planner_feedback: NotRequired[list[str]]
    generated_files: NotRequired[list[str]]
    file_hashes: NotRequired[dict[str, str]]
    coder_result: NotRequired[dict[str, Any]]
    verification_report: NotRequired[dict[str, Any]]
    review_report: NotRequired[dict[str, Any]]
    feedback_history: NotRequired[list[dict[str, Any]]]
    coder_human_decision: NotRequired[dict[str, Any]]
    reviewer_human_decision: NotRequired[dict[str, Any]]
    iteration: NotRequired[int]
    max_iterations: NotRequired[int]
    status: NotRequired[str]
    artifact_path: NotRequired[str | None]
    artifact_hash: NotRequired[str | None]
    errors: NotRequired[list[dict[str, Any]]]
    planner_errors: NotRequired[list[dict[str, Any]]]
