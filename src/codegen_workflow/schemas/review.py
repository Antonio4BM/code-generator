"""Pydantic schemas for reviewer findings and reports."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

FindingSeverity = Literal["blocking", "major", "minor", "suggestion"]

FindingCategory = Literal[
    "requirements",
    "correctness",
    "architecture",
    "security",
    "performance",
    "testing",
    "maintainability",
    "documentation",
]

ReviewVerdict = Literal["approve", "request_changes", "replan"]


class ReviewFinding(BaseModel):
    """A single structured review finding with evidence."""

    finding_id: str = Field(..., min_length=1, description="Stable finding identifier.")
    severity: FindingSeverity = Field(..., description="Impact severity.")
    category: FindingCategory = Field(..., description="Finding taxonomy category.")
    file: str | None = Field(
        default=None,
        description="Relative file path when the finding is file-localized.",
    )
    line: int | None = Field(
        default=None,
        ge=1,
        description="1-based line number when available.",
    )
    description: str = Field(
        ..., min_length=1, description="Concrete defect description."
    )
    evidence: str = Field(
        ..., min_length=1, description="Evidence supporting the finding."
    )
    recommendation: str = Field(
        ...,
        min_length=1,
        description="Actionable recommendation to resolve the finding.",
    )


class ReviewReport(BaseModel):
    """Complete reviewer verdict for one generated candidate project."""

    verdict: ReviewVerdict = Field(..., description="Overall review decision.")
    acceptance_criteria_results: dict[str, bool] = Field(
        default_factory=dict,
        description="Per acceptance-criterion pass/fail keyed by stable id.",
    )
    manifest_results: dict[str, bool] = Field(
        default_factory=dict,
        description="Whether each planned manifest path exists.",
    )
    reviewed_files: list[str] = Field(
        default_factory=list,
        description="Relative paths inspected during the review.",
    )
    findings: list[ReviewFinding] = Field(default_factory=list)
    residual_risks: list[str] = Field(default_factory=list)
    summary: str = Field(..., min_length=1, description="Short review summary.")
