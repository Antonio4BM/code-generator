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
    """Full happy path from ticket through both gates to artifact download."""
    started = client.post("/run-ticket", json={"ticket": "Build a hello CLI"})
    assert started.status_code == 202
    assert started.json()["status"] == "awaiting_coder_approval"
    workflow_id = started.json()["workflow_id"]

    after_coder = client.post(
        f"/runs/{workflow_id}/decision",
        json={"decision": "approve"},
    )
    assert after_coder.status_code == 202
    assert after_coder.json()["status"] == "awaiting_reviewer_approval"

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
    """request_changes re-runs the coder before final approvals."""
    coder = MagicMock(side_effect=mock_coder)
    graph = build_graph(
        checkpointer=InMemorySaver(),
        planner=mock_planner,
        coder=coder,
        reviewer=mock_reviewer,
        verify=mock_verify,
        workspace_base_dir=workspace_root,
    )
    app = create_app(settings=api_settings, graph=graph)
    with TestClient(app) as client:
        started = client.post("/run-ticket", json={"ticket": "Build a hello CLI"})
        workflow_id = started.json()["workflow_id"]
        assert started.json()["status"] == "awaiting_coder_approval"
        assert coder.call_count == 1

        revised = client.post(
            f"/runs/{workflow_id}/decision",
            json={"decision": "request_changes", "feedback": "add README"},
        )
        assert revised.status_code == 202
        assert revised.json()["status"] == "awaiting_coder_approval"
        assert coder.call_count == 2

        after_coder = client.post(
            f"/runs/{workflow_id}/decision",
            json={"decision": "approve"},
        )
        assert after_coder.json()["status"] == "awaiting_reviewer_approval"

        completed = client.post(
            f"/runs/{workflow_id}/decision",
            json={"decision": "approve", "feedback": "final ok"},
        )
        assert completed.json()["status"] == "completed"
        assert completed.json()["artifact_url"]
