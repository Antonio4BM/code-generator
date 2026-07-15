"""Tests for POST /run-ticket."""

from __future__ import annotations

from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from codegen_workflow.api.app import create_app
from codegen_workflow.api.config import APISettings
from codegen_workflow.graph import build_graph
from tests.api.conftest import mock_coder, mock_planner, mock_reviewer, mock_verify


def test_run_ticket_success(client: TestClient) -> None:
    """Valid free-text tickets start a workflow and return a workflow ID."""
    response = client.post("/run-ticket", json={"ticket": "Build a hello CLI"})
    assert response.status_code == 202
    body = response.json()
    assert body["workflow_id"]
    assert body["status"] == "awaiting_coder_approval"
    assert body["interrupt"] is not None
    assert body["interrupt"]["gate"] == "coder"
    assert body["trace_url"].endswith("/trace")


def test_free_text_ticket_accepted(client: TestClient) -> None:
    """Arbitrary free-text tickets are accepted without extra metadata."""
    response = client.post(
        "/run-ticket",
        json={"ticket": "Create a todo REST API in FastAPI with tests"},
    )
    assert response.status_code == 202
    assert response.json()["workflow_id"]


def test_empty_ticket_rejected(client: TestClient) -> None:
    """Empty tickets are rejected by request validation."""
    response = client.post("/run-ticket", json={"ticket": ""})
    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"


def test_whitespace_ticket_rejected(client: TestClient) -> None:
    """Whitespace-only tickets are rejected."""
    response = client.post("/run-ticket", json={"ticket": "   \n\t  "})
    assert response.status_code == 422


def test_excessively_long_ticket_rejected(client: TestClient) -> None:
    """Tickets longer than 20_000 characters are rejected."""
    response = client.post("/run-ticket", json={"ticket": "x" * 20_001})
    assert response.status_code == 422


def test_unknown_request_field_rejected(client: TestClient) -> None:
    """Unknown request fields are rejected (extra=forbid)."""
    response = client.post(
        "/run-ticket",
        json={"ticket": "Build something", "language": "python"},
    )
    assert response.status_code == 422


def test_server_side_workflow_id_generation(client: TestClient) -> None:
    """Clients must not supply a workflow ID; the server generates one."""
    response = client.post("/run-ticket", json={"ticket": "Build a hello CLI"})
    assert response.status_code == 202
    assert "workflow_id" in response.json()
    # Providing workflow_id in body is rejected as unknown field
    rejected = client.post(
        "/run-ticket",
        json={"ticket": "Build a hello CLI", "workflow_id": "client-id"},
    )
    assert rejected.status_code == 422


def test_ticket_mapped_to_user_request(
    api_settings: APISettings,
    workspace_root,
) -> None:
    """HTTP ticket is mapped to the graph's user_request field."""
    captured: dict[str, str] = {}

    def capturing_planner(state):
        captured["user_request"] = state["user_request"]
        return mock_planner(state)

    graph = build_graph(
        checkpointer=InMemorySaver(),
        planner=capturing_planner,
        coder=mock_coder,
        reviewer=mock_reviewer,
        verify=mock_verify,
        workspace_base_dir=workspace_root,
    )
    app = create_app(settings=api_settings, graph=graph)
    with TestClient(app) as client:
        client.post("/run-ticket", json={"ticket": "  Ship a CLI  "})
    assert captured["user_request"] == "Ship a CLI"


def test_optional_bounded_max_iterations(client: TestClient) -> None:
    """Optional max_iterations within 1..4 is accepted."""
    response = client.post(
        "/run-ticket",
        json={"ticket": "Build a hello CLI", "max_iterations": 2},
    )
    assert response.status_code == 202
    workflow_id = response.json()["workflow_id"]
    status = client.get(f"/runs/{workflow_id}")
    assert status.json()["max_iterations"] == 2


def test_invalid_iteration_limits_rejected(client: TestClient) -> None:
    """Out-of-range max_iterations values are rejected."""
    assert client.post(
        "/run-ticket",
        json={"ticket": "Build a hello CLI", "max_iterations": 0},
    ).status_code == 422
    assert client.post(
        "/run-ticket",
        json={"ticket": "Build a hello CLI", "max_iterations": 5},
    ).status_code == 422


def test_response_when_coder_gate_interrupts(client: TestClient) -> None:
    """Coder-gate interrupts return an explicit paused status."""
    body = client.post("/run-ticket", json={"ticket": "Build a hello CLI"}).json()
    assert body["status"] == "awaiting_coder_approval"
    assert body["interrupt"]["gate"] == "coder"
    assert "verification_report" in body["interrupt"]


def test_response_when_workflow_completes(
    api_settings: APISettings,
    workspace_root,
) -> None:
    """Completed runs return status completed and an artifact URL."""
    # Use real mocked graph through two approvals quickly via service
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
        first = client.post("/run-ticket", json={"ticket": "Build a hello CLI"})
        workflow_id = first.json()["workflow_id"]
        second = client.post(
            f"/runs/{workflow_id}/decision",
            json={"decision": "approve", "feedback": ""},
        )
        assert second.json()["status"] == "awaiting_reviewer_approval"
        final = client.post(
            f"/runs/{workflow_id}/decision",
            json={"decision": "approve", "feedback": "ship it"},
        )
        assert final.status_code == 200
        body = final.json()
        assert body["status"] == "completed"
        assert body["artifact_url"] == f"/runs/{workflow_id}/artifact"
        assert body["result"]["status"] == "completed"


def test_max_iterations_reached_response(
    api_settings: APISettings,
    workspace_root,
) -> None:
    """max_iterations_reached is explicit and non-approved without an artifact."""
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
        started = client.post(
            "/run-ticket",
            json={"ticket": "Build a hello CLI", "max_iterations": 1},
        )
        workflow_id = started.json()["workflow_id"]
        # request_changes after first iteration with max=1 ends the loop
        ended = client.post(
            f"/runs/{workflow_id}/decision",
            json={"decision": "request_changes", "feedback": "more tests"},
        )
        body = ended.json()
        assert body["status"] == "max_iterations_reached"
        assert body["artifact_url"] is None
        assert "not approved" in body["message"].lower()


def test_planner_failure_mapping(planner_fail_client: TestClient) -> None:
    """Planner failures map to a structured non-success status."""
    response = planner_fail_client.post(
        "/run-ticket",
        json={"ticket": "Build a hello CLI"},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "planner_failed"
    assert body["artifact_url"] is None
    assert "planner" in body["message"].lower()


def test_handlers_do_not_invoke_agent_nodes_directly(
    api_settings: APISettings,
    workspace_root,
) -> None:
    """API handlers invoke the compiled graph, not Planner/Coder/Reviewer callables."""
    planner = MagicMock(side_effect=mock_planner)
    coder = MagicMock(side_effect=mock_coder)
    reviewer = MagicMock(side_effect=mock_reviewer)
    graph = build_graph(
        checkpointer=InMemorySaver(),
        planner=planner,
        coder=coder,
        reviewer=reviewer,
        verify=mock_verify,
        workspace_base_dir=workspace_root,
    )
    # Spies on graph.invoke — handlers must call this
    original_invoke = graph.invoke
    invoke_spy = MagicMock(side_effect=original_invoke)
    graph.invoke = invoke_spy  # type: ignore[method-assign]

    app = create_app(settings=api_settings, graph=graph)
    with TestClient(app) as client:
        client.post("/run-ticket", json={"ticket": "Build a hello CLI"})

    assert invoke_spy.call_count == 1
    # Nodes run via the graph, not via direct API imports
    assert planner.call_count == 1
    assert coder.call_count == 1
    assert reviewer.call_count == 0
