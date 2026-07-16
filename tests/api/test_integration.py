"""Mocked end-to-end API workflow integration tests."""

from __future__ import annotations

from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from codegen_workflow.api.app import create_app
from codegen_workflow.api.config import APISettings
from codegen_workflow.graph import build_graph
from tests.api.conftest import mock_coder, mock_planner, mock_reviewer, mock_verify


def test_happy_path_to_artifact(client: TestClient) -> None:
    """Full happy path: auto-review, one human approve, then download."""
    started = client.post("/run-ticket", json={"ticket": "Build a hello CLI"})
    assert started.status_code == 202
    assert started.json()["status"] == "awaiting_reviewer_approval"
    assert started.json()["interrupt"]["gate"] == "reviewer"
    workflow_id = started.json()["workflow_id"]

    completed = client.post(
        f"/runs/{workflow_id}/decision",
        json={"decision": "approve", "feedback": "looks good"},
    )
    assert completed.status_code == 200
    assert completed.json()["status"] == "completed"
    assert completed.json()["artifact_url"] == f"/runs/{workflow_id}/artifact"

    artifact = client.get(f"/runs/{workflow_id}/artifact")
    assert artifact.status_code == 200
    assert artifact.content[:2] == b"PK"


def test_revision_path(
    api_settings: APISettings,
    workspace_root,
) -> None:
    """request_changes re-runs planner/coder/review before packaging."""
    coder = MagicMock(side_effect=mock_coder)
    reviewer = MagicMock(side_effect=mock_reviewer)
    graph = build_graph(
        checkpointer=InMemorySaver(),
        planner=mock_planner,
        coder=coder,
        reviewer=reviewer,
        verify=mock_verify,
        workspace_base_dir=workspace_root,
    )
    app = create_app(settings=api_settings, graph=graph)
    with TestClient(app) as client:
        started = client.post("/run-ticket", json={"ticket": "Build a hello CLI"})
        workflow_id = started.json()["workflow_id"]
        assert started.json()["status"] == "awaiting_reviewer_approval"
        assert coder.call_count == 1
        assert reviewer.call_count == 1

        revised = client.post(
            f"/runs/{workflow_id}/decision",
            json={"decision": "request_changes", "feedback": "add README"},
        )
        assert revised.status_code == 202
        assert revised.json()["status"] == "awaiting_reviewer_approval"
        assert coder.call_count == 2
        assert reviewer.call_count == 2

        completed = client.post(
            f"/runs/{workflow_id}/decision",
            json={"decision": "approve", "feedback": "final ok"},
        )
        assert completed.json()["status"] == "completed"
        assert completed.json()["artifact_url"]
