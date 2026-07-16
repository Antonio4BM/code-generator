"""UUID-scoped workspace creation and layout helpers.

Each workflow run owns an isolated directory under ``workspaces/`` with
subfolders for the candidate project, snapshots, reports, and final
packaging output. The coder may write only inside ``candidate/``.

When a LangGraph ``thread_id`` is supplied at compile/runtime, that value
is reused as ``workflow_id`` so checkpoint resumption and workspace paths
stay aligned.
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any

from codegen_workflow.routing import (
    MAX_ITERATIONS,
    STATUS_INVALID_INPUT,
    validate_max_iterations,
)
from codegen_workflow.state import WorkflowState

# Default parent directory for workflow workspaces.
DEFAULT_WORKSPACES_ROOT = Path("workspaces")

# Subdirectories created under every workflow workspace root.
WORKSPACE_SUBDIRS = ("candidate", "snapshots", "reports", "final")


def create_workflow_id() -> str:
    """Generate a new UUID string for a workflow run.

    Returns:
        A UUID4 string used as ``workflow_id`` and preferably as
        ``thread_id``.
    """
    return str(uuid.uuid4())


def workspace_root_for(
    workflow_id: str,
    base_dir: Path | str | None = None,
) -> Path:
    """Resolve the absolute workspace root for a workflow UUID.

    Args:
        workflow_id: Workflow UUID string.
        base_dir: Optional parent directory for all workspaces. Defaults
            to ``workspaces`` under the current working directory.

    Returns:
        Absolute path to ``<base_dir>/<workflow_id>``.
    """
    parent = Path(base_dir) if base_dir is not None else DEFAULT_WORKSPACES_ROOT
    return (parent / workflow_id).resolve()


def create_workspace(
    workflow_id: str | None = None,
    base_dir: Path | str | None = None,
) -> tuple[str, Path]:
    """Create an isolated workspace layout for one workflow run.

    Args:
        workflow_id: Optional UUID. A new UUID is generated when omitted.
        base_dir: Optional parent directory for workspaces.

    Returns:
        A tuple of ``(workflow_id, workspace_path)``.

    Raises:
        FileExistsError: If the workspace directory already exists.
    """
    wf_id = workflow_id or create_workflow_id()
    root = workspace_root_for(wf_id, base_dir=base_dir)
    if root.exists():
        raise FileExistsError(f"Workspace already exists: {root}")
    root.mkdir(parents=True, exist_ok=False)
    for name in WORKSPACE_SUBDIRS:
        (root / name).mkdir(parents=True, exist_ok=False)
    return wf_id, root


def candidate_dir(workspace_path: Path | str) -> Path:
    """Return the coder-writable candidate directory.

    Args:
        workspace_path: Absolute or relative workspace root.

    Returns:
        Path to the ``candidate`` subdirectory.
    """
    return Path(workspace_path).resolve() / "candidate"


def snapshots_dir(workspace_path: Path | str) -> Path:
    """Return the snapshots directory for a workspace.

    Args:
        workspace_path: Absolute or relative workspace root.

    Returns:
        Path to the ``snapshots`` subdirectory.
    """
    return Path(workspace_path).resolve() / "snapshots"


def reports_dir(workspace_path: Path | str) -> Path:
    """Return the reports directory for a workspace.

    Args:
        workspace_path: Absolute or relative workspace root.

    Returns:
        Path to the ``reports`` subdirectory.
    """
    return Path(workspace_path).resolve() / "reports"


def final_dir(workspace_path: Path | str) -> Path:
    """Return the final packaging directory for a workspace.

    Args:
        workspace_path: Absolute or relative workspace root.

    Returns:
        Path to the ``final`` subdirectory.
    """
    return Path(workspace_path).resolve() / "final"


def snapshot_candidate(
    workspace_path: Path | str,
    iteration: int,
) -> Path:
    """Copy the candidate tree into ``snapshots/iteration_<n>``.

    Args:
        workspace_path: Workspace root containing ``candidate/``.
        iteration: Iteration number used in the snapshot folder name.

    Returns:
        Path to the created snapshot directory.
    """
    source = candidate_dir(workspace_path)
    destination = snapshots_dir(workspace_path) / f"iteration_{iteration}"
    if destination.exists():
        shutil.rmtree(destination)
    if source.exists():
        shutil.copytree(source, destination)
    else:
        destination.mkdir(parents=True, exist_ok=True)
    return destination


def initialize_workspace_node(
    state: WorkflowState,
    *,
    base_dir: Path | str | None = None,
    workflow_id: str | None = None,
) -> dict[str, Any]:
    """LangGraph node that creates a UUID-scoped workspace.

    Generates a workflow UUID (or reuses the runnable ``thread_id``),
    creates the isolated directory layout, and initializes iteration
    counters and feedback history.

    Args:
        state: Incoming workflow state. Only ``user_request`` is required.
        base_dir: Optional override for the workspaces parent directory.
        workflow_id: Optional ID. Prefer aligning this with the LangGraph
            ``thread_id`` so checkpoints and workspace paths match.

    Returns:
        State update containing workspace identifiers and defaults, or a
        terminal ``invalid_input`` update when the request is empty.
    """
    user_request = (state.get("user_request") or "").strip()
    if not user_request:
        return {
            "user_request": state.get("user_request") or "",
            "status": STATUS_INVALID_INPUT,
            "errors": [
                {
                    "type": STATUS_INVALID_INPUT,
                    "message": "user_request is required and must be non-empty",
                }
            ],
            "feedback_history": [],
            "iteration": 0,
            "max_iterations": MAX_ITERATIONS,
            "artifact_path": None,
            "artifact_hash": None,
        }

    raw_max = state.get("max_iterations")
    if raw_max is None:
        max_iterations = MAX_ITERATIONS
    else:
        try:
            max_iterations = validate_max_iterations(int(raw_max))
        except (TypeError, ValueError) as exc:
            return {
                "user_request": user_request,
                "status": STATUS_INVALID_INPUT,
                "errors": [
                    {
                        "type": STATUS_INVALID_INPUT,
                        "message": str(exc),
                    }
                ],
                "feedback_history": [],
                "iteration": 0,
                "max_iterations": MAX_ITERATIONS,
                "artifact_path": None,
                "artifact_hash": None,
            }

    wf_id, workspace_path = create_workspace(
        workflow_id=workflow_id,
        base_dir=base_dir,
    )

    return {
        "user_request": user_request,
        "workflow_id": wf_id,
        "workspace_path": str(workspace_path),
        "plan": {},
        "previous_plan": {},
        "change_request": {},
        "plan_diff": {},
        "planner_feedback": [],
        "generated_files": [],
        "file_hashes": {},
        "coder_result": {},
        "verification_report": {},
        "review_report": {},
        "feedback_history": [],
        "coder_human_decision": {},
        "reviewer_human_decision": {},
        "iteration": 0,
        "max_iterations": max_iterations,
        "status": "planning",
        "artifact_path": None,
        "artifact_hash": None,
        "errors": [],
        "planner_errors": [],
    }
