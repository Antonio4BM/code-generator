"""Tests for deterministic manifest revision helpers."""

from __future__ import annotations

from codegen_workflow.revision import ManifestDiff, compare_manifests, normalize_manifest_path


def _plan(paths: list[str]) -> dict:
    return {
        "file_manifest": [
            {
                "path": path,
                "purpose": "test",
                "file_type": "source",
                "requirements": ["exists"],
                "depends_on": [],
            }
            for path in paths
        ]
    }


def test_added_removed_retained() -> None:
    """Added, removed, and retained paths are computed correctly."""
    previous = _plan(["src/app.py", "src/legacy.py", "README.md"])
    revised = _plan(["src/app.py", "src/auth.py", "README.md"])
    diff = compare_manifests(previous, revised)
    assert diff.added == ["src/auth.py"]
    assert diff.removed == ["src/legacy.py"]
    assert diff.retained == ["README.md", "src/app.py"]


def test_empty_diff() -> None:
    """Identical manifests produce empty added/removed lists."""
    plan = _plan(["a.py", "b.py"])
    diff = compare_manifests(plan, plan)
    assert diff.added == []
    assert diff.removed == []
    assert diff.retained == ["a.py", "b.py"]


def test_path_normalization() -> None:
    """Backslashes normalize to POSIX paths."""
    assert normalize_manifest_path(r"src\\auth.py") == "src/auth.py"
    previous = {"file_manifest": [{"path": r"src\\app.py", "purpose": "x"}]}
    revised = {"file_manifest": [{"path": "src/app.py", "purpose": "x"}]}
    diff = compare_manifests(previous, revised)
    assert diff.added == []
    assert diff.removed == []
    assert diff.retained == ["src/app.py"]


def test_deterministic_sorting() -> None:
    """Diff lists are sorted for deterministic comparisons."""
    previous = _plan(["z.py", "a.py"])
    revised = _plan(["m.py", "b.py", "a.py"])
    diff = compare_manifests(previous, revised)
    assert diff.added == ["b.py", "m.py"]
    assert diff.removed == ["z.py"]
    assert diff.retained == ["a.py"]


def test_completely_replaced_manifest() -> None:
    """Full replacements mark every old path removed and every new path added."""
    previous = _plan(["old/a.py", "old/b.py"])
    revised = _plan(["new/a.py", "new/b.py"])
    diff = compare_manifests(previous, revised)
    assert diff.added == ["new/a.py", "new/b.py"]
    assert diff.removed == ["old/a.py", "old/b.py"]
    assert diff.retained == []


def test_project_plan_instances() -> None:
    """ProjectPlan-shaped dicts are accepted by compare_manifests."""
    previous = _plan(["src/app.py", "README.md"])
    revised = _plan(["src/app.py", "README.md", "src/auth.py"])
    diff = compare_manifests(previous, revised)
    assert isinstance(diff, ManifestDiff)
    assert diff.added == ["src/auth.py"]
    assert diff.removed == []
    assert diff.retained == ["README.md", "src/app.py"]
