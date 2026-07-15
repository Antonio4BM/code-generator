"""Tests for POST /runs/{workflow_id}/decision."""

from __future__ import annotations

from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from codegen_workflow.api.app import create_app
from codegen_workflow.api.config import APISettings
from codegen_workflow.graph import build_graph, run_config_for_thread
from tests.api.conftest import mock_coder, mock_planner, mock_reviewer, mock_verify


def _start(client: TestClient) -> str:
    """Start a workflow and return its ID."""
    response = client.post("/run-ticket", json={"ticket": "Build a hello CLI"})
    assert response.status_code == 202
    return response.json()["workflow_id"]


def test_human_approval_resumption(client: TestClient) -> None:
    """Approving at the coder gate resumes into the reviewer interrupt."""
    workflow_id = _start(client)
    response = client.post(
        f"/runs/{workflow_id}/decision",
        json={"decision": "approve", "feedback": ""},
    )
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "awaiting_reviewer_approval"
    assert body["interrupt"]["gate"] == "reviewer"


def test_human_change_request_resumption(
    api_settings: APISettings,
    workspace_root,
) -> None:
    """request_changes resumes the same thread and re-runs the coder."""
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
        workflow_id = _start(client)
        assert coder.call_count == 1
        response = client.post(
            f"/runs/{workflow_id}/decision",
            json={"decision": "request_changes", "feedback": "add README details"},
        )
        assert response.status_code == 202
        assert response.json()["status"] == "awaiting_coder_approval"
        assert coder.call_count == 2


def test_human_replan_resumption(
    api_settings: APISettings,
    workspace_root,
) -> None:
    """replan resumes the same thread and re-invokes the planner."""
    planner = MagicMock(side_effect=mock_planner)
    graph = build_graph(
        checkpointer=InMemorySaver(),
        planner=planner,
        coder=mock_coder,
        reviewer=mock_reviewer,
        verify=mock_verify,
        workspace_base_dir=workspace_root,
    )
    app = create_app(settings=api_settings, graph=graph)
    with TestClient(app) as client:
        workflow_id = _start(client)
        assert planner.call_count == 1
        response = client.post(
            f"/runs/{workflow_id}/decision",
            json={"decision": "replan", "feedback": "prefer a library package"},
        )
        assert response.status_code == 202
        assert planner.call_count == 2
        assert response.json()["interrupt"]["gate"] == "coder"


def test_human_abort_resumption(client: TestClient) -> None:
    """abort terminates the workflow without an approved artifact."""
    workflow_id = _start(client)
    response = client.post(
        f"/runs/{workflow_id}/decision",
        json={"decision": "abort", "feedback": "stop"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "aborted"
    assert body["artifact_url"] is None


def test_feedback_required_for_change_requests(client: TestClient) -> None:
    """request_changes and replan require meaningful feedback."""
    workflow_id = _start(client)
    assert (
        client.post(
            f"/runs/{workflow_id}/decision",
            json={"decision": "request_changes", "feedback": ""},
        ).status_code
        == 422
    )
    assert (
        client.post(
            f"/runs/{workflow_id}/decision",
            json={"decision": "replan", "feedback": "   "},
        ).status_code
        == 422
    )


def test_unknown_workflow_decision(client: TestClient) -> None:
    """Decisions for unknown workflows return 404."""
    response = client.post(
        "/runs/00000000-0000-0000-0000-000000000000/decision",
        json={"decision": "approve"},
    )
    assert response.status_code == 404
    assert response.json()["code"] == "workflow_not_found"


def test_decision_on_completed_workflow(client: TestClient) -> None:
    """A completed workflow cannot accept another decision."""
    workflow_id = _start(client)
    client.post(
        f"/runs/{workflow_id}/decision",
        json={"decision": "approve"},
    )
    client.post(
        f"/runs/{workflow_id}/decision",
        json={"decision": "approve", "feedback": "final"},
    )
    response = client.post(
        f"/runs/{workflow_id}/decision",
        json={"decision": "approve"},
    )
    assert response.status_code == 409


def test_decision_when_no_interrupt(
    api_settings: APISettings,
    workspace_root,
) -> None:
    """Decisions are rejected when the workflow is not paused."""
    # Force a terminal planner failure (no interrupt)
    def failing_planner(state):
        return {
            "plan": {},
            "planner_errors": [{"type": "planner_failed", "message": "x"}],
            "status": "planner_failed",
        }

    graph = build_graph(
        checkpointer=InMemorySaver(),
        planner=failing_planner,
        coder=mock_coder,
        reviewer=mock_reviewer,
        verify=mock_verify,
        workspace_base_dir=workspace_root,
    )
    app = create_app(settings=api_settings, graph=graph)
    with TestClient(app) as client:
        started = client.post("/run-ticket", json={"ticket": "Build a hello CLI"})
        workflow_id = started.json()["workflow_id"]
        response = client.post(
            f"/runs/{workflow_id}/decision",
            json={"decision": "approve"},
        )
        assert response.status_code == 409


def test_duplicate_decision_protection(
    api_settings: APISettings,
    workspace_root,
) -> None:
    """Concurrent resume attempts for the same interrupt are rejected."""
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
        workflow_id = _start(client)
        service = app.state.workflow_service
        lock = service._lock_for(workflow_id)
        assert lock.acquire(blocking=False)
        try:
            response = client.post(
                f"/runs/{workflow_id}/decision",
                json={"decision": "approve"},
            )
            assert response.status_code == 409
            assert response.json()["code"] == "duplicate_decision"
        finally:
            lock.release()


def test_resume_uses_same_thread_id(
    api_settings: APISettings,
    workspace_root,
) -> None:
    """Human decisions resume the original LangGraph thread_id."""
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
        workflow_id = _start(client)
        client.post(
            f"/runs/{workflow_id}/decision",
            json={"decision": "approve"},
        )
        snapshot = graph.get_state(run_config_for_thread(workflow_id))
        assert snapshot.values["workflow_id"] == workflow_id
        # Interrupt still attached to the same thread
        assert snapshot.interrupts
        assert snapshot.interrupts[0].value["gate"] == "reviewer"


def test_response_when_reviewer_gate_interrupts(client: TestClient) -> None:
    """Reviewer-gate interrupts return an explicit paused status."""
    workflow_id = _start(client)
    response = client.post(
        f"/runs/{workflow_id}/decision",
        json={"decision": "approve"},
    )
    body = response.json()
    assert body["status"] == "awaiting_reviewer_approval"
    assert body["interrupt"]["gate"] == "reviewer"
    assert "reviewer_verdict" in body["interrupt"]


def test_handlers_do_not_duplicate_graph_routing(
    api_settings: APISettings,
    workspace_root,
) -> None:
    """Handlers resume via Command and do not select successor nodes."""
    graph = build_graph(
        checkpointer=InMemorySaver(),
        planner=mock_planner,
        coder=mock_coder,
        reviewer=mock_reviewer,
        verify=mock_verify,
        workspace_base_dir=workspace_root,
    )
    original = graph.invoke
    spy = MagicMock(side_effect=original)
    graph.invoke = spy  # type: ignore[method-assign]
    app = create_app(settings=api_settings, graph=graph)
    with TestClient(app) as client:
        workflow_id = _start(client)
        client.post(
            f"/runs/{workflow_id}/decision",
            json={"decision": "approve"},
        )
    # Second invoke is a resume Command — not a new graph_input dict
    resume_call = spy.call_args_list[1]
    assert isinstance(resume_call.args[0], Command)
