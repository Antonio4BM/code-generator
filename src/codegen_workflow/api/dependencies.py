"""FastAPI dependency providers for the workflow API."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import Depends, Request

from codegen_workflow.api.config import APISettings
from codegen_workflow.api.service import WorkflowService


def get_settings(request: Request) -> APISettings:
    """Return application settings from the FastAPI app state.

    Args:
        request: Incoming HTTP request.

    Returns:
        Process-wide :class:`APISettings`.
    """
    return request.app.state.settings


def get_workflow_service(request: Request) -> WorkflowService:
    """Return the shared workflow service from app state.

    Args:
        request: Incoming HTTP request.

    Returns:
        Configured :class:`WorkflowService`.
    """
    return request.app.state.workflow_service


def get_graph(request: Request) -> Any:
    """Return the compiled graph from app state.

    Args:
        request: Incoming HTTP request.

    Returns:
        Compiled LangGraph instance.
    """
    return request.app.state.graph


SettingsDep = Annotated[APISettings, Depends(get_settings)]
WorkflowDep = Annotated[WorkflowService, Depends(get_workflow_service)]
GraphDep = Annotated[Any, Depends(get_graph)]
