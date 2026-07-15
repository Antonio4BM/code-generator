"""API exception types and structured error responses."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class APIError(BaseModel):
    """Structured error payload returned to API clients.

    Attributes:
        code: Stable machine-readable error code.
        message: Safe human-readable explanation.
        workflow_id: Related workflow identifier when applicable.
        details: Optional list of non-sensitive detail strings.
    """

    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    workflow_id: str | None = None
    details: list[str] | None = Field(default=None)


class APIErrorDetail(Exception):
    """Base exception carrying an HTTP status and structured payload."""

    def __init__(
        self,
        *,
        status_code: int,
        code: str,
        message: str,
        workflow_id: str | None = None,
        details: list[str] | None = None,
    ) -> None:
        """Initialize the exception.

        Args:
            status_code: HTTP status code for the response.
            code: Machine-readable error code.
            message: Safe client-facing message.
            workflow_id: Optional related workflow identifier.
            details: Optional non-sensitive detail strings.
        """
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.workflow_id = workflow_id
        self.details = details

    def to_model(self) -> APIError:
        """Convert to the public error schema.

        Returns:
            An :class:`APIError` instance.
        """
        return APIError(
            code=self.code,
            message=self.message,
            workflow_id=self.workflow_id,
            details=self.details,
        )


class WorkflowNotFoundError(APIErrorDetail):
    """Raised when a workflow ID does not map to a persisted run."""

    def __init__(self, workflow_id: str) -> None:
        super().__init__(
            status_code=404,
            code="workflow_not_found",
            message="The requested workflow does not exist.",
            workflow_id=workflow_id,
        )


class InvalidWorkflowTransitionError(APIErrorDetail):
    """Raised when a decision or artifact request violates workflow state."""

    def __init__(
        self,
        workflow_id: str,
        message: str,
        *,
        code: str = "invalid_workflow_transition",
        details: list[str] | None = None,
    ) -> None:
        super().__init__(
            status_code=409,
            code=code,
            message=message,
            workflow_id=workflow_id,
            details=details,
        )


class InvalidHumanDecisionError(APIErrorDetail):
    """Raised when a human decision payload is semantically invalid."""

    def __init__(
        self,
        message: str,
        *,
        workflow_id: str | None = None,
        details: list[str] | None = None,
    ) -> None:
        super().__init__(
            status_code=400,
            code="invalid_human_decision",
            message=message,
            workflow_id=workflow_id,
            details=details,
        )


class ArtifactNotFoundError(APIErrorDetail):
    """Raised when no artifact exists for a completed workflow."""

    def __init__(self, workflow_id: str) -> None:
        super().__init__(
            status_code=404,
            code="artifact_not_found",
            message="No artifact is available for this workflow.",
            workflow_id=workflow_id,
        )


class ArtifactNotReadyError(APIErrorDetail):
    """Raised when an artifact is requested before completion."""

    def __init__(self, workflow_id: str) -> None:
        super().__init__(
            status_code=409,
            code="artifact_not_ready",
            message="The workflow has not completed; artifact is not available.",
            workflow_id=workflow_id,
        )


class ServiceUnavailableError(APIErrorDetail):
    """Raised when required infrastructure is unavailable."""

    def __init__(self, message: str, *, details: list[str] | None = None) -> None:
        super().__init__(
            status_code=503,
            code="service_unavailable",
            message=message,
            details=details,
        )


class WorkflowTimeoutError(APIErrorDetail):
    """Raised when a bounded graph invocation exceeds its timeout."""

    def __init__(self, workflow_id: str | None = None) -> None:
        super().__init__(
            status_code=504,
            code="workflow_timeout",
            message="The workflow operation timed out.",
            workflow_id=workflow_id,
        )


class GraphInvocationError(APIErrorDetail):
    """Raised when graph execution fails unexpectedly but safely."""

    def __init__(
        self,
        message: str = "Workflow execution failed.",
        *,
        workflow_id: str | None = None,
        details: list[str] | None = None,
    ) -> None:
        super().__init__(
            status_code=500,
            code="graph_invocation_failed",
            message=message,
            workflow_id=workflow_id,
            details=details,
        )


def error_body(exc: APIErrorDetail) -> dict[str, Any]:
    """Serialize an API exception to a JSON-compatible dict.

    Args:
        exc: Structured API exception.

    Returns:
        Dictionary matching :class:`APIError`.
    """
    return exc.to_model().model_dump(exclude_none=True)
