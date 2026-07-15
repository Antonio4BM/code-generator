"""Workflow create, status, decision, and trace endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Response, status

from codegen_workflow.api.dependencies import WorkflowDep
from codegen_workflow.api.errors import APIError
from codegen_workflow.api.schemas import (
    HumanDecisionRequest,
    HumanDecisionResponse,
    RunStatusResponse,
    RunTicketRequest,
    RunTicketResponse,
    RunTraceResponse,
)

router = APIRouter(tags=["workflows"])


@router.post(
    "/run-ticket",
    response_model=RunTicketResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Start a code-generation workflow",
    responses={
        201: {"description": "Workflow created and finished a terminal step"},
        202: {"description": "Workflow created and paused for human review"},
        422: {"description": "Request validation failed"},
        500: {"model": APIError, "description": "Unexpected workflow failure"},
        504: {"model": APIError, "description": "Workflow timed out"},
    },
)
async def run_ticket(
    request: RunTicketRequest,
    workflow: WorkflowDep,
    response: Response,
) -> RunTicketResponse:
    """Start a workflow from a free-text software ticket.

    Args:
        request: Validated ticket and optional iteration configuration.
        workflow: Injected compiled LangGraph workflow service.
        response: Mutable response used to set 201/202.

    Returns:
        The workflow identifier, current status, and result or interrupt.
    """
    result, code = await workflow.start_ticket(
        ticket=request.ticket,
        max_iterations=request.max_iterations,
    )
    response.status_code = code
    return result


@router.post(
    "/runs/{workflow_id}/decision",
    response_model=HumanDecisionResponse,
    summary="Resume a workflow with a human decision",
    responses={
        200: {"description": "Decision applied; workflow continued or finished"},
        202: {"description": "Decision applied; workflow paused again"},
        400: {"model": APIError, "description": "Invalid human decision"},
        404: {"model": APIError, "description": "Workflow not found"},
        409: {"model": APIError, "description": "Invalid workflow transition"},
    },
)
async def submit_decision(
    workflow_id: str,
    request: HumanDecisionRequest,
    workflow: WorkflowDep,
    response: Response,
) -> HumanDecisionResponse:
    """Resume an interrupted workflow with the same thread identifier.

    Args:
        workflow_id: Existing workflow / LangGraph thread ID.
        request: Human decision and optional feedback.
        workflow: Injected workflow service.
        response: Mutable response used to set 200/202.

    Returns:
        Updated workflow status, interrupt, or final result.
    """
    result, code = await workflow.submit_decision(workflow_id, request)
    response.status_code = code
    return HumanDecisionResponse.model_validate(result.model_dump())


@router.get(
    "/runs/{workflow_id}",
    response_model=RunStatusResponse,
    summary="Get workflow run status",
    responses={
        404: {"model": APIError, "description": "Workflow not found"},
    },
)
async def get_run_status(
    workflow_id: str,
    workflow: WorkflowDep,
) -> RunStatusResponse:
    """Return the current persisted state of a workflow.

    Args:
        workflow_id: Workflow identifier.
        workflow: Injected workflow service.

    Returns:
        Status, iteration counters, pending gate, and artifact URL.
    """
    return workflow.get_status(workflow_id)


@router.get(
    "/runs/{workflow_id}/trace",
    response_model=RunTraceResponse,
    summary="Get workflow intermediate trace",
    responses={
        404: {"model": APIError, "description": "Workflow not found"},
    },
)
async def get_run_trace(
    workflow_id: str,
    workflow: WorkflowDep,
) -> RunTraceResponse:
    """Return redacted intermediate decisions for a workflow.

    Args:
        workflow_id: Workflow identifier.
        workflow: Injected workflow service.

    Returns:
        Ordered list of client-safe trace events.
    """
    return workflow.get_trace(workflow_id)
