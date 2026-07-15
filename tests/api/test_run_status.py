"""Tests for GET /runs/{workflow_id}."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_run_status_retrieval(client: TestClient) -> None:
    """Status endpoint returns persisted counters and pending gate."""
    created = client.post("/run-ticket", json={"ticket": "Build a hello CLI"})
    workflow_id = created.json()["workflow_id"]
    response = client.get(f"/runs/{workflow_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["workflow_id"] == workflow_id
    assert body["status"] == "awaiting_coder_approval"
    assert body["pending_gate"] == "coder"
    assert body["iteration"] == 1
    assert body["max_iterations"] >= 1
    assert "hello.py" in body["generated_files"]
    assert body["artifact_url"] is None


def test_unknown_workflow_status(client: TestClient) -> None:
    """Unknown workflow IDs return 404."""
    response = client.get("/runs/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404
    assert response.json()["code"] == "workflow_not_found"


def test_status_is_read_only(client: TestClient) -> None:
    """Repeated status requests do not modify workflow state."""
    created = client.post("/run-ticket", json={"ticket": "Build a hello CLI"})
    workflow_id = created.json()["workflow_id"]
    first = client.get(f"/runs/{workflow_id}").json()
    second = client.get(f"/runs/{workflow_id}").json()
    assert first == second
