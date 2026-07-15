"""Pydantic request and response schemas for the HTTP API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

RunStatus = Literal[
    "planning",
    "coding",
    "verifying",
    "awaiting_coder_approval",
    "reviewing",
    "awaiting_reviewer_approval",
    "completed",
    "aborted",
    "max_iterations_reached",
    "planner_failed",
    "coder_failed",
    "verification_failed",
    "reviewer_failed",
    "packaging_failed",
    "invalid_input",
]

PAUSED_STATUSES: frozenset[str] = frozenset(
    {
        "awaiting_coder_approval",
        "awaiting_reviewer_approval",
    }
)

TERMINAL_FAILURE_STATUSES: frozenset[str] = frozenset(
    {
        "aborted",
        "max_iterations_reached",
        "planner_failed",
        "coder_failed",
        "verification_failed",
        "reviewer_failed",
        "packaging_failed",
        "invalid_input",
    }
)

TERMINAL_STATUSES: frozenset[str] = TERMINAL_FAILURE_STATUSES | frozenset({"completed"})


class RunTicketRequest(BaseModel):
    """Request body for starting a new code-generation workflow."""

    model_config = ConfigDict(extra="forbid")

    ticket: str = Field(
        min_length=1,
        max_length=20_000,
        description="Free-text software feature or project request.",
    )
    max_iterations: int | None = Field(
        default=None,
        ge=1,
        le=4,
        description="Optional bounded number of implementation attempts.",
    )

    @field_validator("ticket")
    @classmethod
    def ticket_must_not_be_whitespace(cls, value: str) -> str:
        """Reject whitespace-only tickets after strip.

        Args:
            value: Raw ticket text.

        Returns:
            Stripped ticket text.

        Raises:
            ValueError: If the ticket is empty after stripping.
        """
        stripped = value.strip()
        if not stripped:
            raise ValueError("ticket must not be empty or whitespace-only")
        return stripped


class RunTicketResponse(BaseModel):
    """Response returned after creating or resuming a workflow."""

    model_config = ConfigDict(extra="forbid")

    workflow_id: str
    status: RunStatus
    message: str
    interrupt: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    artifact_url: str | None = None
    trace_url: str


class HumanDecisionRequest(BaseModel):
    """Human gate decision used to resume a paused workflow."""

    model_config = ConfigDict(extra="forbid")

    decision: Literal["approve", "request_changes", "replan", "abort"]
    feedback: str = Field(default="", max_length=10_000)

    @model_validator(mode="after")
    def require_feedback_for_revision(self) -> HumanDecisionRequest:
        """Require meaningful feedback for change and replan decisions.

        Returns:
            The validated model instance.

        Raises:
            ValueError: If feedback is missing for revision decisions.
        """
        if self.decision in {"request_changes", "replan"}:
            if not self.feedback.strip():
                raise ValueError(
                    f"feedback is required when decision is {self.decision!r}"
                )
        return self


class HumanDecisionResponse(RunTicketResponse):
    """Response returned after submitting a human decision."""


class RunStatusResponse(BaseModel):
    """Current persisted status for a workflow run."""

    model_config = ConfigDict(extra="forbid")

    workflow_id: str
    status: RunStatus
    iteration: int
    max_iterations: int
    pending_gate: str | None
    generated_files: list[str]
    artifact_url: str | None
    created_at: str | None
    updated_at: str | None


class TraceEvent(BaseModel):
    """One redacted, client-safe workflow trace event."""

    model_config = ConfigDict(extra="forbid")

    sequence: int
    node: str
    status: str
    iteration: int | None
    summary: str
    timestamp: str | None
    details: dict[str, Any] | None = None


class RunTraceResponse(BaseModel):
    """Ordered trace of intermediate workflow decisions and outcomes."""

    model_config = ConfigDict(extra="forbid")

    workflow_id: str
    events: list[TraceEvent]


class CandidateFileTreeResponse(BaseModel):
    """Relative paths of files under the workflow candidate workspace."""

    model_config = ConfigDict(extra="forbid")

    workflow_id: str
    files: list[str]


class CandidateFileContentResponse(BaseModel):
    """UTF-8 contents of one file under the candidate workspace."""

    model_config = ConfigDict(extra="forbid")

    workflow_id: str
    path: str
    content: str
    encoding: Literal["utf-8"] = "utf-8"
    size_bytes: int


class HealthResponse(BaseModel):
    """Liveness probe response."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"]


class ReadyResponse(BaseModel):
    """Readiness probe response."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ready", "not_ready"]
    checks: dict[str, bool]
