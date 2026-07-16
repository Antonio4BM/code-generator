"""Shared typed state for the code-generation workflow.

The graph stores structured artifacts explicitly. A message list is not
used as the primary state representation.

State invariants:

- ``workflow_id`` and ``workspace_path`` are immutable after initialization.
- ``iteration`` increments only when the coder completes an attempt.
- ``feedback_history`` is append-only.
- Agent nodes return partial state updates only.
- ``plan`` is always the authoritative complete project plan.
- ``previous_plan`` / ``plan_diff`` describe the latest planner revision
  and stay available through verification; a new ``change_request``
  clears stale revision fields before the next planner revision.
"""

from __future__ import annotations

from typing import Any, TypedDict


class WorkflowState(TypedDict, total=False):
    """Typed LangGraph state for one user software request.

    Attributes:
        user_request: Plain-text software requirement from the caller.
        workflow_id: UUID identifying this workflow run (aligned with
            the LangGraph ``thread_id`` when practical).
        workspace_path: Absolute path to the isolated workspace root.
        plan: Validated project plan produced by the planner.
        previous_plan: Authoritative plan before the latest revision.
        change_request: Latest human-requested structural modification.
        plan_diff: Deterministic manifest diff for the latest revision.
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
    workflow_id: str
    workspace_path: str
    plan: dict[str, Any]
    previous_plan: dict[str, Any]
    change_request: dict[str, Any]
    plan_diff: dict[str, Any]
    planner_feedback: list[str]
    generated_files: list[str]
    file_hashes: dict[str, str]
    coder_result: dict[str, Any]
    verification_report: dict[str, Any]
    review_report: dict[str, Any]
    feedback_history: list[dict[str, Any]]
    coder_human_decision: dict[str, Any]
    reviewer_human_decision: dict[str, Any]
    iteration: int
    max_iterations: int
    status: str
    artifact_path: str | None
    artifact_hash: str | None
    errors: list[dict[str, Any]]
    planner_errors: list[dict[str, Any]]
