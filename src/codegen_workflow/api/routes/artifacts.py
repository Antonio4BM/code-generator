"""Artifact download endpoint."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import FileResponse

from codegen_workflow.api.dependencies import WorkflowDep
from codegen_workflow.api.errors import APIError

router = APIRouter(tags=["artifacts"])


@router.get(
    "/runs/{workflow_id}/artifact",
    summary="Download the approved workflow artifact",
    responses={
        404: {"model": APIError, "description": "Workflow or artifact not found"},
        409: {"model": APIError, "description": "Artifact not ready"},
    },
)
async def download_artifact(workflow_id: str, workflow: WorkflowDep) -> FileResponse:
    """Serve the final approved ZIP for a completed workflow.

    Args:
        workflow_id: Workflow identifier (also LangGraph thread ID).
        workflow: Injected workflow service.

    Returns:
        ZIP file response with a safe download filename.

    Raises:
        APIErrorDetail: Mapped by application exception handlers.
    """
    artifact_path = workflow.resolve_artifact_path(workflow_id)
    return FileResponse(
        path=artifact_path,
        media_type="application/zip",
        filename=f"{workflow_id}.zip",
    )
