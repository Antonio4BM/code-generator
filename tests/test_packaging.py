"""Tests for ZIP packaging and archive hashing."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

from codegen_workflow.packaging import (
    create_zip_archive,
    iter_packaged_files,
    package_project_node,
    should_exclude,
)
from codegen_workflow.routing import STATUS_PACKAGING_FAILED
from codegen_workflow.workspace import create_workspace


def _write_candidate(workspace: Path) -> None:
    """Populate a candidate tree with source and excluded artifacts."""
    candidate = workspace / "candidate"
    (candidate / "src").mkdir(parents=True)
    (candidate / "src" / "app.py").write_text("print('hi')\n", encoding="utf-8")
    (candidate / "README.md").write_text("# demo\n", encoding="utf-8")
    (candidate / ".env").write_text("SECRET=1\n", encoding="utf-8")
    (candidate / ".env.local").write_text("SECRET=2\n", encoding="utf-8")
    (candidate / "credentials.json").write_text("{}", encoding="utf-8")
    (candidate / "tls.pem").write_text("-----BEGIN-----\n", encoding="utf-8")
    (candidate / "id.key").write_text("key\n", encoding="utf-8")
    venv_dir = candidate / ".venv" / "lib"
    venv_dir.mkdir(parents=True)
    (venv_dir / "sitecustomize.py").write_text("# venv\n", encoding="utf-8")
    cache = candidate / "__pycache__"
    cache.mkdir()
    (cache / "app.cpython-312.pyc").write_bytes(b"\0")
    reports = candidate / "reports"
    reports.mkdir()
    (reports / "tmp.json").write_text("{}\n", encoding="utf-8")


def test_should_exclude_secrets_and_caches(tmp_path: Path) -> None:
    """Secret and temporary paths must be excluded from packaging."""
    root = tmp_path / "candidate"
    root.mkdir()
    assert should_exclude(root / ".env", root) is True
    assert should_exclude(root / ".env.production", root) is True
    assert should_exclude(root / "credentials.json", root) is True
    assert should_exclude(root / "cert.pem", root) is True
    assert should_exclude(root / "private.key", root) is True
    assert should_exclude(root / "__pycache__" / "x.pyc", root) is True
    assert should_exclude(root / ".venv" / "bin" / "python", root) is True
    assert should_exclude(root / "reports" / "out.json", root) is True
    assert should_exclude(root / "node_modules" / "pkg" / "index.js", root) is True
    assert should_exclude(root / "src" / "app.py", root) is False


def test_zip_packaging_and_hash(tmp_path: Path) -> None:
    """Packaging writes a ZIP and returns a matching SHA-256 digest."""
    workflow_id, workspace = create_workspace(base_dir=tmp_path)
    _write_candidate(workspace)
    archive = workspace / "final" / f"{workflow_id}.zip"

    digest = create_zip_archive(workspace / "candidate", archive)

    assert archive.is_file()
    expected = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert digest == expected

    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
    assert "src/app.py" in names
    assert "README.md" in names
    assert ".env" not in names
    assert ".env.local" not in names
    assert "credentials.json" not in names
    assert "tls.pem" not in names
    assert "id.key" not in names
    assert not any(name.startswith(".venv/") for name in names)
    assert not any("__pycache__" in name for name in names)
    assert not any(name.startswith("reports/") for name in names)


def test_package_project_node_updates_state(tmp_path: Path) -> None:
    """The packaging node returns completed status and artifact metadata."""
    workflow_id, workspace = create_workspace(base_dir=tmp_path)
    _write_candidate(workspace)

    update = package_project_node(
        {
            "workflow_id": workflow_id,
            "workspace_path": str(workspace),
            "generated_files": ["src/app.py", "README.md"],
            "reviewer_human_decision": {"decision": "approve", "feedback": ""},
        }
    )

    assert update["status"] == "completed"
    assert update["artifact_path"].endswith(f"{workflow_id}.zip")
    assert len(update["artifact_hash"]) == 64
    assert Path(update["artifact_path"]).is_file()


def test_packaging_requires_final_approval(tmp_path: Path) -> None:
    """Packaging without reviewer approval fails explicitly."""
    workflow_id, workspace = create_workspace(base_dir=tmp_path)
    _write_candidate(workspace)

    update = package_project_node(
        {
            "workflow_id": workflow_id,
            "workspace_path": str(workspace),
            "reviewer_human_decision": {"decision": "request_changes"},
        }
    )
    assert update["status"] == STATUS_PACKAGING_FAILED
    assert update["artifact_path"] is None
    assert update["errors"][0]["type"] == STATUS_PACKAGING_FAILED


def test_iter_packaged_files_skips_excluded(tmp_path: Path) -> None:
    """Iterator yields only non-excluded candidate files."""
    workflow_id, workspace = create_workspace(base_dir=tmp_path)
    del workflow_id
    _write_candidate(workspace)
    relative = [
        path.relative_to(workspace / "candidate").as_posix()
        for path in iter_packaged_files(workspace / "candidate")
    ]
    assert "src/app.py" in relative
    assert ".env" not in relative
    assert "tls.pem" not in relative
