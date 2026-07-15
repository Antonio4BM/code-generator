"""Tests for health, readiness, OpenAPI, CORS, and error sanitization."""

from __future__ import annotations

import logging

from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from codegen_workflow.api.app import create_app
from codegen_workflow.api.config import APISettings
from codegen_workflow.api.logging_config import redact_secrets
from codegen_workflow.api.service import WorkflowService
from codegen_workflow.graph import build_graph
from tests.api.conftest import mock_coder, mock_planner, mock_reviewer, mock_verify


def test_liveness_endpoint(client: TestClient) -> None:
    """GET /health returns a lightweight ok payload."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readiness_endpoint(client: TestClient) -> None:
    """GET /ready verifies dependencies without calling an LLM."""
    response = client.get("/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["checks"]["graph_compiled"] is True
    assert body["checks"]["workspace_writable"] is True


def test_openapi_schema_generation(client: TestClient) -> None:
    """OpenAPI schema documents the primary workflow endpoints."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "/run-ticket" in paths
    assert "/runs/{workflow_id}/decision" in paths
    assert "/runs/{workflow_id}" in paths
    assert "/runs/{workflow_id}/trace" in paths
    assert "/runs/{workflow_id}/artifact" in paths
    assert "/health" in paths
    assert "/ready" in paths


def test_cors_configuration(client: TestClient) -> None:
    """Configured origins receive CORS headers."""
    response = client.options(
        "/run-ticket",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_unexpected_exception_sanitization(
    api_settings: APISettings,
    workspace_root,
) -> None:
    """Unexpected exceptions return a generic message without a stack trace."""
    graph = build_graph(
        checkpointer=InMemorySaver(),
        planner=mock_planner,
        coder=mock_coder,
        reviewer=mock_reviewer,
        verify=mock_verify,
        workspace_base_dir=workspace_root,
    )
    service = WorkflowService(graph, api_settings, checkpointer=InMemorySaver())

    async def boom(*_args, **_kwargs):
        raise RuntimeError("secret stack OPENAI_API_KEY=sk-secretvalue")

    service.start_ticket = boom  # type: ignore[method-assign]
    app = create_app(
        settings=api_settings,
        graph=graph,
        workflow_service=service,
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/run-ticket", json={"ticket": "Build a hello CLI"})
    assert response.status_code == 500
    body = response.json()
    assert body["code"] == "internal_error"
    assert "Traceback" not in response.text
    assert "sk-secretvalue" not in response.text


def test_model_timeout_mapping(
    api_settings: APISettings,
    workspace_root,
) -> None:
    """Timeouts map to HTTP 504 with a structured error payload."""
    graph = build_graph(
        checkpointer=InMemorySaver(),
        planner=mock_planner,
        coder=mock_coder,
        reviewer=mock_reviewer,
        verify=mock_verify,
        workspace_base_dir=workspace_root,
    )
    settings = APISettings(
        openai_api_key=api_settings.openai_api_key,
        workspace_base_dir=api_settings.workspace_base_dir,
        artifact_base_dir=api_settings.artifact_base_dir,
        log_level="WARNING",
        allowed_origins=api_settings.allowed_origins,
        app_env="test",
        workflow_timeout_seconds=0.01,
    )
    service = WorkflowService(graph, settings, checkpointer=InMemorySaver())

    def slow_invoke(*_args, **_kwargs):
        import time

        time.sleep(0.05)
        return {}

    graph.invoke = slow_invoke  # type: ignore[method-assign]
    app = create_app(settings=settings, graph=graph, workflow_service=service)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/run-ticket", json={"ticket": "Build a hello CLI"})
    assert response.status_code == 504
    assert response.json()["code"] == "workflow_timeout"


def test_sensitive_values_are_not_logged(caplog) -> None:
    """Logging helpers redact API keys and authorization material."""
    raw = "Authorization: Bearer sk-abcdefghijklmnop OPENAI_API_KEY=sk-zzzz"
    assert "sk-abcdefghijklmnop" not in redact_secrets(raw)
    assert "sk-zzzz" not in redact_secrets(raw)
    with caplog.at_level(logging.INFO):
        logging.getLogger("codegen_workflow.api").info(
            "safe %s",
            redact_secrets("token=sk-should-hide"),
        )
    assert "sk-should-hide" not in caplog.text


def test_api_keys_loaded_from_environment(monkeypatch, tmp_path) -> None:
    """API settings read secrets from environment variables only."""
    monkeypatch.setenv("OPENAI_API_KEY", "env-secret-key")
    monkeypatch.setenv("WORKSPACE_BASE_DIR", str(tmp_path / "ws"))
    monkeypatch.setenv("ARTIFACT_BASE_DIR", str(tmp_path / "art"))
    monkeypatch.setenv("APP_ENV", "development")
    settings = APISettings.from_env()
    assert settings.openai_api_key == "env-secret-key"


def test_chat_interface_served(client: TestClient) -> None:
    """GET / serves the one-page ticket interface."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Code Generator" in response.text
    assert "/run-ticket" in response.text


def test_failure_mappings_via_status(
    api_settings: APISettings,
    workspace_root,
) -> None:
    """Coder, verification, reviewer, and packaging failures map cleanly."""

    class FakeService(WorkflowService):
        async def start_ticket(self, ticket, max_iterations=None):  # type: ignore[override]
            from codegen_workflow.api.schemas import RunTicketResponse

            response = RunTicketResponse(
                workflow_id="fake-id",
                status=ticket,  # type: ignore[arg-type]
                message=f"{ticket} mapped",
                interrupt=None,
                result={"status": ticket},
                artifact_url=None,
                trace_url="/runs/fake-id/trace",
            )
            return response, 201

    graph = build_graph(
        checkpointer=InMemorySaver(),
        planner=mock_planner,
        coder=mock_coder,
        reviewer=mock_reviewer,
        verify=mock_verify,
        workspace_base_dir=workspace_root,
    )
    for status_name in (
        "coder_failed",
        "verification_failed",
        "reviewer_failed",
        "packaging_failed",
    ):
        service = FakeService(graph, api_settings, checkpointer=InMemorySaver())
        app = create_app(
            settings=api_settings,
            graph=graph,
            workflow_service=service,
        )
        with TestClient(app) as client:
            response = client.post("/run-ticket", json={"ticket": status_name})
            assert response.status_code == 201
            assert response.json()["status"] == status_name
            assert response.json()["artifact_url"] is None
