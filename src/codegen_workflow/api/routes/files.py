"""Candidate workspace file listing and preview endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Query

from codegen_workflow.api.dependencies import WorkflowDep
from codegen_workflow.api.errors import APIError
from codegen_workflow.api.schemas import (
    CandidateFileContentResponse,
    CandidateFileTreeResponse,
)

router = APIRouter(tags=["files"])


@router.get(
    "/runs/{workflow_id}/files",
    response_model=CandidateFileTreeResponse,
    summary="List generated candidate files",
    responses={
        404: {"model": APIError, "description": "Workflow not found"},
        409: {"model": APIError, "description": "Workspace not ready"},
    },
)
async def list_candidate_files(
    workflow_id: str,
    workflow: WorkflowDep,
) -> CandidateFileTreeResponse:
    """Return the relative file tree under the workflow candidate directory.

    Args:
        workflow_id: Existing workflow / LangGraph thread ID.
        workflow: Injected workflow service.

    Returns:
        Sorted relative paths for files under ``candidate/``.
    """
    return workflow.list_candidate_files(workflow_id)


@router.get(
    "/runs/{workflow_id}/files/content",
    response_model=CandidateFileContentResponse,
    summary="Read one generated candidate file",
    responses={
        400: {"model": APIError, "description": "Unreadable or oversized file"},
        404: {"model": APIError, "description": "Workflow or file not found"},
        409: {"model": APIError, "description": "Workspace not ready or path violation"},
    },
)
async def read_candidate_file(
    workflow_id: str,
    workflow: WorkflowDep,
    path: str = Query(
        ...,
        min_length=1,
        max_length=1024,
        description="Relative path under candidate/ (for example src/app.py).",
    ),
) -> CandidateFileContentResponse:
    """Return UTF-8 contents of one file inside the candidate workspace.

    Paths are constrained to the workflow ``candidate/`` directory. Absolute
    paths, ``..`` traversal, and symlink escapes are rejected.

    Args:
        workflow_id: Existing workflow / LangGraph thread ID.
        workflow: Injected workflow service.
        path: Relative path under ``candidate/``.

    Returns:
        File path, UTF-8 content, and size metadata.
    """
    return workflow.read_candidate_file(workflow_id, path)
