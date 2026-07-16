"""Deterministic plan-revision helpers for manifest comparison.

The LLM proposes revised plans; this module calculates structural
differences between previous and revised file manifests without model
involvement.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from codegen_workflow.schemas.plan import ProjectPlan


class ManifestDiff(BaseModel):
    """Structural difference between two plan file manifests.

    Attributes:
        added: Paths present only in the revised plan.
        removed: Paths present only in the previous plan.
        retained: Paths present in both plans.
    """

    model_config = ConfigDict(extra="forbid")

    added: list[str] = Field(default_factory=list)
    removed: list[str] = Field(default_factory=list)
    retained: list[str] = Field(default_factory=list)


def normalize_manifest_path(path: str) -> str:
    """Normalize a manifest path to a relative POSIX path.

    Args:
        path: Raw path from a plan file specification.

    Returns:
        Forward-slash path with surrounding whitespace removed and
        redundant separators collapsed.
    """
    cleaned = str(path or "").strip().replace("\\", "/")
    while "//" in cleaned:
        cleaned = cleaned.replace("//", "/")
    return cleaned.lstrip("./")


def manifest_paths(plan: ProjectPlan | dict[str, Any]) -> set[str]:
    """Extract the set of normalized file-manifest paths from a plan.

    Args:
        plan: ``ProjectPlan`` instance or plan dictionary.

    Returns:
        Set of normalized relative paths.
    """
    if isinstance(plan, ProjectPlan):
        specs = plan.file_manifest
        paths: set[str] = set()
        for spec in specs:
            normalized = normalize_manifest_path(spec.path)
            if normalized:
                paths.add(normalized)
        return paths

    paths = set()
    for entry in plan.get("file_manifest") or []:
        if isinstance(entry, dict):
            normalized = normalize_manifest_path(str(entry.get("path") or ""))
        else:
            normalized = normalize_manifest_path(str(getattr(entry, "path", "") or ""))
        if normalized:
            paths.add(normalized)
    return paths


def compare_manifests(
    previous_plan: ProjectPlan | dict[str, Any],
    revised_plan: ProjectPlan | dict[str, Any],
) -> ManifestDiff:
    """Compare previous and revised plan manifests deterministically.

    Args:
        previous_plan: Plan that was authoritative before revision.
        revised_plan: Newly validated authoritative plan.

    Returns:
        Sorted ``added``, ``removed``, and ``retained`` path lists.
    """
    previous = manifest_paths(previous_plan)
    revised = manifest_paths(revised_plan)
    return ManifestDiff(
        added=sorted(revised - previous),
        removed=sorted(previous - revised),
        retained=sorted(previous & revised),
    )


def plan_diff_payload(
    previous_plan: ProjectPlan | dict[str, Any],
    revised_plan: ProjectPlan | dict[str, Any],
) -> dict[str, Any]:
    """Build a JSON-serializable plan_diff state value.

    Args:
        previous_plan: Plan before revision.
        revised_plan: Plan after revision.

    Returns:
        Dictionary suitable for ``WorkflowState["plan_diff"]``.
    """
    return compare_manifests(previous_plan, revised_plan).model_dump()
