"""Tests for GET /runs/{workflow_id}/trace."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_trace_retrieval(client: TestClient) -> None:
    """Trace endpoint exposes intermediate workflow milestones."""
    created = client.post("/run-ticket", json={"ticket": "Build a hello CLI"})
    workflow_id = created.json()["workflow_id"]
    response = client.get(f"/runs/{workflow_id}/trace")
    assert response.status_code == 200
    body = response.json()
    assert body["workflow_id"] == workflow_id
    nodes = [event["node"] for event in body["events"]]
    assert "initialize_workspace" in nodes
    assert "planner" in nodes
    assert "coder" in nodes
    assert "verify" in nodes
    assert any("human_gate" in node for node in nodes)


def test_trace_redaction_of_sensitive_information(client: TestClient) -> None:
    """Trace details must not expose secrets or prompt contents."""
    created = client.post("/run-ticket", json={"ticket": "Build a hello CLI"})
    workflow_id = created.json()["workflow_id"]
    body = client.get(f"/runs/{workflow_id}/trace").json()
    serialized = str(body).lower()
    assert "sk-" not in serialized
    assert "openai_api_key" not in serialized
    assert "authorization" not in serialized
    for event in body["events"]:
        details = event.get("details") or {}
        for key in details:
            assert "prompt" not in key.lower()
            assert "secret" not in key.lower()
            assert "api_key" not in key.lower()


def test_unknown_workflow_trace(client: TestClient) -> None:
    """Unknown workflow traces return 404."""
    response = client.get("/runs/00000000-0000-0000-0000-000000000000/trace")
    assert response.status_code == 404
