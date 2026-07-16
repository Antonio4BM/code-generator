"""Tests for GET /runs/{workflow_id}/artifact."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from codegen_workflow.api.app import create_app
from codegen_workflow.api.config import APISettings
from codegen_workflow.api.errors import InvalidWorkflowTransitionError
from codegen_workflow.api.service import WorkflowService
from codegen_workflow.graph import build_graph
from langgraph.checkpoint.memory import InMemorySaver

from tests.api.conftest import mock_coder, mock_planner, mock_reviewer, mock_verify


def _complete(client: TestClient) -> str:
    """Drive a mocked workflow to completion and return its ID."""
    started = client.post("/run-ticket", json={"ticket": "Build a hello CLI"})
    workflow_id = started.json()["workflow_id"]
    final = client.post(
        f"/runs/{workflow_id}/decision",
        json={"decision": "approve", "feedback": "done"},
    )
    assert final.json()["status"] == "completed"
    return workflow_id


def test_artifact_download_after_completion(client: TestClient) -> None:
    """Completed workflows serve a ZIP artifact."""
    workflow_id = _complete(client)
    response = client.get(f"/runs/{workflow_id}/artifact")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/zip")
    assert response.content[:2] == b"PK"


def test_artifact_rejection_before_completion(client: TestClient) -> None:
    """Artifacts are unavailable while the workflow is still running/paused."""
    started = client.post("/run-ticket", json={"ticket": "Build a hello CLI"})
    workflow_id = started.json()["workflow_id"]
    response = client.get(f"/runs/{workflow_id}/artifact")
    assert response.status_code == 409
    assert response.json()["code"] == "artifact_not_ready"


def test_artifact_path_containment_enforcement(
    api_settings: APISettings,
    workspace_root: Path,
) -> None:
    """Artifact paths outside configured roots are rejected."""
    graph = build_graph(
        checkpointer=InMemorySaver(),
        planner=mock_planner,
        coder=mock_coder,
        reviewer=mock_reviewer,
        verify=mock_verify,
        workspace_base_dir=workspace_root,
    )
    app = create_app(settings=api_settings, graph=graph)
    with TestClient(app) as client:
        workflow_id = _complete(client)
        service: WorkflowService = app.state.workflow_service
        snapshot = service.graph.get_state(
            {"configurable": {"thread_id": workflow_id}}
        )
        # Point state at an escaped path
        escaped = Path("/tmp/evil.zip")
        escaped.write_bytes(b"PK\x03\x04evil")
        try:
            service.graph.update_state(
                {"configurable": {"thread_id": workflow_id}},
                {"artifact_path": str(escaped), "status": "completed"},
            )
            try:
                service.resolve_artifact_path(workflow_id)
                raise AssertionError("expected path containment failure")
            except InvalidWorkflowTransitionError as exc:
                assert exc.code == "artifact_path_violation"
        finally:
            escaped.unlink(missing_ok=True)


def test_unknown_workflow_artifact(client: TestClient) -> None:
    """Unknown workflow artifact requests return 404."""
    response = client.get("/runs/00000000-0000-0000-0000-000000000000/artifact")
    assert response.status_code == 404
