"""FastAPI application factory for the code-generation workflow API."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from langgraph.checkpoint.memory import InMemorySaver
from starlette.middleware.base import BaseHTTPMiddleware

from codegen_workflow.api.config import APISettings
from codegen_workflow.api.errors import APIError, APIErrorDetail, error_body
from codegen_workflow.api.logging_config import (
    configure_logging,
    log_extra,
    new_request_id,
    redact_secrets,
)
from codegen_workflow.api.routes import artifacts, files, health, runs
from codegen_workflow.api.service import WorkflowService
from codegen_workflow.graph import create_workflow

logger = logging.getLogger("codegen_workflow.api")

_STATIC_DIR = Path(__file__).resolve().parent / "static"


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Attach a correlation ID to each request and response."""

    async def dispatch(self, request: Request, call_next):  # type: ignore[no-untyped-def]
        """Process the request with a correlation identifier.

        Args:
            request: Incoming HTTP request.
            call_next: Downstream ASGI callable.

        Returns:
            Downstream response with ``X-Request-ID`` header.
        """
        request_id = request.headers.get("X-Request-ID") or new_request_id()
        request.state.request_id = request_id
        logger.info(
            "request_start method=%s",
            request.method,
            extra=log_extra(
                request_id=request_id,
                endpoint=request.url.path,
            ),
        )
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "request_complete status=%s",
            response.status_code,
            extra=log_extra(
                request_id=request_id,
                endpoint=request.url.path,
            ),
        )
        return response


def create_app(
    *,
    settings: APISettings | None = None,
    graph: Any | None = None,
    checkpointer: Any | None = None,
    workflow_service: WorkflowService | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        settings: Optional preloaded settings (tests inject these).
        graph: Optional prebuilt compiled graph.
        checkpointer: Optional checkpointer instance.
        workflow_service: Optional prebuilt workflow service.

    Returns:
        Configured :class:`FastAPI` application.
    """
    settings = settings or APISettings.from_env()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Initialize application resources on startup.

        Args:
            app: FastAPI application instance.

        Yields:
            Control while the application serves requests.
        """
        local_checkpointer = checkpointer or InMemorySaver()
        local_graph = graph or create_workflow(
            checkpointer=local_checkpointer,
            workspace_base_dir=settings.workspace_base_dir,
        )
        settings.workspace_base_dir.mkdir(parents=True, exist_ok=True)
        settings.artifact_base_dir.mkdir(parents=True, exist_ok=True)
        app.state.settings = settings
        app.state.checkpointer = local_checkpointer
        app.state.graph = local_graph
        app.state.workflow_service = workflow_service or WorkflowService(
            local_graph,
            settings,
            checkpointer=local_checkpointer,
        )
        yield

    app = FastAPI(
        title="Code Generation Workflow API",
        description=(
            "HTTP adapter around the planner-coder-reviewer LangGraph workflow. "
            "Submit a free-text ticket, approve human gates, inspect traces, "
            "and download the approved ZIP artifact."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # When callers inject a graph/service at construction time (tests), seed
    # state immediately so dependency resolution works without waiting for
    # lifespan in contexts that already manage it.
    if graph is not None or workflow_service is not None:
        local_checkpointer = checkpointer or InMemorySaver()
        local_graph = graph or create_workflow(
            checkpointer=local_checkpointer,
            workspace_base_dir=settings.workspace_base_dir,
        )
        app.state.settings = settings
        app.state.checkpointer = local_checkpointer
        app.state.graph = local_graph
        app.state.workflow_service = workflow_service or WorkflowService(
            local_graph,
            settings,
            checkpointer=local_checkpointer,
        )

    app.add_middleware(RequestIdMiddleware)

    origins = settings.allowed_origins
    if origins:
        if settings.is_production and "*" in origins:
            logger.warning("Rejecting permissive CORS '*' in production")
            origins = [o for o in origins if o != "*"]
        if origins:
            app.add_middleware(
                CORSMiddleware,
                allow_origins=origins,
                allow_credentials=True,
                allow_methods=["GET", "POST", "OPTIONS"],
                allow_headers=["Content-Type", "X-Request-ID", "Idempotency-Key"],
            )

    app.include_router(health.router)
    app.include_router(runs.router)
    app.include_router(artifacts.router)
    app.include_router(files.router)

    if _STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def chat_interface() -> HTMLResponse:
        """Serve the one-page chat ticket interface.

        Returns:
            HTML page that submits tickets to ``POST /run-ticket``.
        """
        index = _STATIC_DIR / "index.html"
        if index.is_file():
            return HTMLResponse(index.read_text(encoding="utf-8"))
        return HTMLResponse(
            "<!doctype html><html><body><h1>Code Generator</h1>"
            "<p>API is running. POST /run-ticket to start.</p></body></html>"
        )

    @app.exception_handler(APIErrorDetail)
    async def api_error_handler(request: Request, exc: APIErrorDetail) -> JSONResponse:
        """Map structured API errors to JSON responses.

        Args:
            request: Current request.
            exc: Structured API exception.

        Returns:
            JSON error response.
        """
        request_id = getattr(request.state, "request_id", "-")
        logger.warning(
            "api_error code=%s",
            exc.code,
            extra=log_extra(
                request_id=request_id,
                workflow_id=exc.workflow_id,
                endpoint=request.url.path,
            ),
        )
        return JSONResponse(status_code=exc.status_code, content=error_body(exc))

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        """Return structured validation errors without leaking internals.

        Args:
            request: Current request.
            exc: Pydantic / FastAPI validation error.

        Returns:
            JSON 422 error response.
        """
        details = []
        for err in exc.errors():
            loc = ".".join(str(part) for part in err.get("loc", ()))
            details.append(f"{loc}: {err.get('msg')}")
        body = APIError(
            code="validation_error",
            message="Request validation failed.",
            details=details,
        ).model_dump(exclude_none=True)
        return JSONResponse(status_code=422, content=body)

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        """Sanitize unexpected exceptions.

        Args:
            request: Current request.
            exc: Unexpected exception.

        Returns:
            Generic JSON 500 response without a stack trace.
        """
        request_id = getattr(request.state, "request_id", "-")
        logger.exception(
            "unhandled_exception type=%s detail=%s",
            type(exc).__name__,
            redact_secrets(str(exc)),
            extra=log_extra(
                request_id=request_id,
                endpoint=request.url.path,
                error_type=type(exc).__name__,
            ),
        )
        body = APIError(
            code="internal_error",
            message="An unexpected error occurred.",
        ).model_dump(exclude_none=True)
        return JSONResponse(status_code=500, content=body)

    return app


app = create_app()
