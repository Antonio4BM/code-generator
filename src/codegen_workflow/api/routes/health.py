"""Liveness and readiness endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Response, status

from codegen_workflow.api.dependencies import WorkflowDep
from codegen_workflow.api.schemas import HealthResponse, ReadyResponse

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
)
async def health() -> HealthResponse:
    """Return a lightweight liveness response.

    Returns:
        ``{\"status\": \"ok\"}`` when the process is alive.
    """
    return HealthResponse(status="ok")


@router.get(
    "/ready",
    response_model=ReadyResponse,
    summary="Readiness probe",
    responses={
        503: {"description": "Service is not ready"},
    },
)
async def ready(workflow: WorkflowDep, response: Response) -> ReadyResponse:
    """Verify critical dependencies without invoking an LLM.

    Args:
        workflow: Injected workflow service.
        response: Mutable FastAPI response for status-code adjustment.

    Returns:
        Readiness status and per-check results.
    """
    checks = workflow.readiness_checks()
    is_ready = all(checks.values())
    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadyResponse(status="not_ready", checks=checks)
    return ReadyResponse(status="ready", checks=checks)
