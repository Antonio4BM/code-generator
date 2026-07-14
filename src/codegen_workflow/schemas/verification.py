"""Pydantic schemas for deterministic verification reports."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class CommandResult(BaseModel):
    """Result of a single allowlisted verification command.

    Attributes:
        name: Logical step name (for example ``lint`` or ``test``).
        command: Exact argv executed for the step.
        exit_code: Process exit code.
        stdout: Captured standard output.
        stderr: Captured standard error.
        duration_seconds: Wall-clock duration for the command.
        skipped: Whether the step was skipped by configuration.
    """

    name: str
    command: list[str] = Field(default_factory=list)
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    duration_seconds: float = 0.0
    skipped: bool = False


class VerificationReport(BaseModel):
    """Aggregated verification outcome for a candidate workspace.

    Attributes:
        passed: True when every executed check exits successfully.
        overall_status: High-level status string for routing displays.
        commands: Ordered list of per-command results.
        errors: Structured verification errors, if any.
        metadata: Optional extra diagnostic fields.
    """

    passed: bool = False
    overall_status: Literal["passed", "failed", "error", "skipped"] = "failed"
    commands: list[CommandResult] = Field(default_factory=list)
    errors: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
