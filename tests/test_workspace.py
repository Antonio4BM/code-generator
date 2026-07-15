"""Tests for workspace initialization and layout."""

from __future__ import annotations

from pathlib import Path

from codegen_workflow.routing import MAX_ITERATIONS, STATUS_INVALID_INPUT
from codegen_workflow.workspace import (
    create_workflow_id,
    create_workspace,
    initialize_workspace_node,
)


def test_uuid_generation() -> None:
    """Workflow IDs are unique UUID strings."""
    first = create_workflow_id()
    second = create_workflow_id()
    assert first != second
    assert len(first) == 36


def test_workspace_creation(tmp_path: Path) -> None:
    """create_workspace builds the required subdirectory layout."""
    workflow_id, root = create_workspace(base_dir=tmp_path)
    assert root.is_dir()
    assert workflow_id in str(root)
    for name in ("candidate", "snapshots", "reports", "final"):
        assert (root / name).is_dir()


def test_initialize_initial_state_values(tmp_path: Path) -> None:
    """Initialization sets counters, empty collections, and planning status."""
    update = initialize_workspace_node(
        {"user_request": "Build a CLI"},
        base_dir=tmp_path,
        workflow_id="init-state-1",
    )
    assert update["status"] == "planning"
    assert update["iteration"] == 0
    assert update["max_iterations"] == MAX_ITERATIONS
    assert update["feedback_history"] == []
    assert update["errors"] == []
    assert update["generated_files"] == []
    assert update["workflow_id"] == "init-state-1"
    assert Path(update["workspace_path"]).is_dir()


def test_empty_request_rejection(tmp_path: Path) -> None:
    """Empty user_request yields an explicit invalid_input terminal update."""
    update = initialize_workspace_node({"user_request": "   "}, base_dir=tmp_path)
    assert update["status"] == STATUS_INVALID_INPUT
    assert update["errors"]
    assert update["errors"][0]["type"] == STATUS_INVALID_INPUT
    assert "workspace_path" not in update


def test_invalid_max_iterations_rejected(tmp_path: Path) -> None:
    """Out-of-range max_iterations is rejected as invalid_input."""
    update = initialize_workspace_node(
        {"user_request": "demo", "max_iterations": 99},
        base_dir=tmp_path,
    )
    assert update["status"] == STATUS_INVALID_INPUT
