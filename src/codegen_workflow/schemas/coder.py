"""Pydantic schemas for coder node results."""

from __future__ import annotations

from pydantic import BaseModel, Field


class CoderResult(BaseModel):
    """Structured summary of one coder iteration.

    Attributes:
        summary: Human-readable description of work performed.
        created_files: Relative paths created in this iteration.
        modified_files: Relative paths updated in this iteration.
        deleted_files: Relative paths removed in this iteration.
        unresolved_issues: Findings or problems still outstanding.
        feedback_resolutions: Map of finding id to resolution description.
        manifest_compliance: Whether each planned manifest path exists.
    """

    summary: str = Field(..., description="Short summary of the coder iteration.")
    created_files: list[str] = Field(default_factory=list)
    modified_files: list[str] = Field(default_factory=list)
    deleted_files: list[str] = Field(default_factory=list)
    unresolved_issues: list[str] = Field(default_factory=list)
    feedback_resolutions: dict[str, str] = Field(default_factory=dict)
    manifest_compliance: dict[str, bool] = Field(default_factory=dict)
