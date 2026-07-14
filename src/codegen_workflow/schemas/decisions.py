"""Pydantic schemas for human-gate decisions."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

DecisionLiteral = Literal["approve", "request_changes", "replan", "abort"]


class HumanDecision(BaseModel):
    """Decision returned by a human approval gate.

    Attributes:
        decision: Routing choice selected by the human reviewer.
        feedback: Free-text rationale or change requests from the human.
    """

    decision: DecisionLiteral = Field(
        ...,
        description="Routing choice after inspecting workflow artifacts.",
    )
    feedback: str = Field(
        default="",
        description="Optional human feedback appended to history.",
    )
