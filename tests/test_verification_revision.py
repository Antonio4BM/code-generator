"""Tests for plan-revision structural verification helpers."""

from __future__ import annotations

from pathlib import Path

from codegen_workflow.nodes.verification import (
    build_verification_report,
    evaluate_manifest_revision,
)


def test_added_file_present_passes_structural_check() -> None:
    """Added files that exist do not fail structural revision checks."""
    report = evaluate_manifest_revision(
        {"src/app.py", "src/auth.py"},
        {},
        {"added": ["src/auth.py"], "removed": [], "retained": ["src/app.py"]},
    )
    assert report is not None
    assert report["missing_added"] == []
    assert report["successfully_added"] == ["src/auth.py"]


def test_added_file_missing_fails_structural_check() -> None:
    """Missing added files are reported."""
    report = evaluate_manifest_revision(
        {"src/app.py"},
        {},
        {"added": ["src/auth.py"], "removed": [], "retained": ["src/app.py"]},
    )
    assert report is not None
    assert report["missing_added"] == ["src/auth.py"]


def test_removed_file_deleted_passes_structural_check() -> None:
    """Removed files absent from the workspace pass."""
    report = evaluate_manifest_revision(
        {"src/app.py"},
        {},
        {"added": [], "removed": ["src/payments.py"], "retained": ["src/app.py"]},
    )
    assert report is not None
    assert report["still_present"] == []
    assert report["successfully_removed"] == ["src/payments.py"]


def test_removed_file_still_present_fails_structural_check() -> None:
    """Removed files that remain on disk fail."""
    report = evaluate_manifest_revision(
        {"src/app.py", "src/payments.py"},
        {},
        {"added": [], "removed": ["src/payments.py"], "retained": ["src/app.py"]},
    )
    assert report is not None
    assert report["still_present"] == ["src/payments.py"]


def test_no_structural_change_skips_manifest_revision(tmp_path: Path) -> None:
    """Empty plan diffs keep prior command-only verification behavior."""
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "hello.py").write_text("print('ok')\n", encoding="utf-8")

    def runner(name, argv, cwd):
        from codegen_workflow.schemas.verification import CommandResult

        return CommandResult(
            name=name,
            command=list(argv),
            exit_code=0,
            stdout="ok\n",
            stderr="",
            duration_seconds=0.01,
            skipped=False,
        )

    report = build_verification_report(
        tmp_path,
        plan={
            "language": "python",
            "install_commands": [],
            "validation_commands": ["python3", "-c", "print('ok')"],
        },
        plan_diff={"added": [], "removed": [], "retained": ["hello.py"]},
        command_runner=runner,
    )
    assert report.passed is True
    assert "manifest_revision" not in report.metadata
