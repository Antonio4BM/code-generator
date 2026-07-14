"""UUID-scoped workspace creation and layout helpers.

Each workflow run owns an isolated directory under ``workspaces/`` with
subfolders for the candidate project, snapshots, reports, and final
packaging output. The coder may write only inside ``candidate/``.
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any

from codegen_workflow.routing import MAX_ITERATIONS
from codegen_workflow.state import WorkflowState

# Default parent directory for workflow workspaces.
DEFAULT_WORKSPACES_ROOT = Path("workspaces")

# Subdirectories created under every workflow workspace root.
WORKSPACE_SUBDIRS = ("candidate", "snapshots", "reports", "final")


def create_workflow_id() -> str:
    """Generate a new UUID string for a workflow run.

    Returns:
        A UUID4 string used as ``workflow_id`` and ``thread_id``.
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
) -> dict[str, Any]:
    """LangGraph node that creates a UUID-scoped workspace.

    Generates a workflow UUID, creates the isolated directory layout,
    and initializes iteration counters and feedback history.

    Args:
        state: Incoming workflow state. Only ``user_request`` is required.
        base_dir: Optional override for the workspaces parent directory.

    Returns:
        State update containing workspace identifiers and defaults.

    Raises:
        ValueError: If ``user_request`` is missing or blank.
    """
    user_request = (state.get("user_request") or "").strip()
    if not user_request:
        raise ValueError("user_request is required and must be non-empty")

    workflow_id, workspace_path = create_workspace(base_dir=base_dir)
    max_iterations = int(state.get("max_iterations") or MAX_ITERATIONS)

    return {
        "user_request": user_request,
        "workflow_id": workflow_id,
        "workspace_path": str(workspace_path),
        "plan": {},
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
