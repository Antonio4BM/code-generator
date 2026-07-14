"""Package an approved candidate workspace into a ZIP artifact.

After final human approval the packaging step snapshots the candidate
project, excludes secrets and temporary artifacts, writes a ZIP archive,
and records a SHA-256 digest for provenance.
"""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path
from typing import Any, Iterable

from codegen_workflow.state import WorkflowState
from codegen_workflow.workspace import candidate_dir, final_dir

# Directory and file name patterns excluded from packaged archives.
EXCLUDED_DIR_NAMES = frozenset(
    {
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".git",
        "node_modules",
        ".tox",
        ".eggs",
        "dist",
        "build",
        ".idea",
        ".vscode",
    }
)

EXCLUDED_FILE_NAMES = frozenset(
    {
        ".env",
        ".env.local",
        ".env.production",
        "credentials.json",
        "secrets.json",
        ".DS_Store",
    }
)

EXCLUDED_SUFFIXES = frozenset(
    {
        ".pyc",
        ".pyo",
        ".pyd",
        ".so",
        ".egg-info",
    }
)


def should_exclude(path: Path, root: Path) -> bool:
    """Return whether a path must be omitted from the archive.

    Args:
        path: Absolute path under consideration.
        root: Absolute candidate root used for relative checks.

    Returns:
        True when the path matches an exclusion rule.
    """
    try:
        relative = path.relative_to(root)
    except ValueError:
        return True

    for part in relative.parts:
        if part in EXCLUDED_DIR_NAMES:
            return True
        if part.endswith(".egg-info"):
            return True

    name = path.name
    if name in EXCLUDED_FILE_NAMES:
        return True
    if name.startswith(".env"):
        return True
    if path.suffix in EXCLUDED_SUFFIXES:
        return True
    return False


def iter_packaged_files(candidate_root: Path) -> Iterable[Path]:
    """Yield files under ``candidate_root`` that should be archived.

    Args:
        candidate_root: Root of the approved candidate project.

    Yields:
        Absolute paths of included files.
    """
    if not candidate_root.exists():
        return
    for path in sorted(candidate_root.rglob("*")):
        if not path.is_file():
            continue
        if should_exclude(path, candidate_root):
            continue
        yield path


def sha256_file(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file.

    Args:
        path: File to hash.

    Returns:
        Lower-case hexadecimal SHA-256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_zip_archive(
    candidate_root: Path,
    archive_path: Path,
) -> str:
    """Create a ZIP archive of an approved candidate tree.

    Args:
        candidate_root: Directory containing approved generated files.
        archive_path: Destination path for the ``.zip`` file.

    Returns:
        SHA-256 hex digest of the written archive.
    """
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in iter_packaged_files(candidate_root):
            arcname = file_path.relative_to(candidate_root).as_posix()
            zf.write(file_path, arcname=arcname)
    return sha256_file(archive_path)


def package_project_node(state: WorkflowState) -> dict[str, Any]:
    """LangGraph node that packages the approved workspace.

    Copies metadata about the archive into workflow state. Virtual
    environments, caches, secrets, and temporary reports are excluded.

    Args:
        state: Workflow state after final human approval.

    Returns:
        State update with ``artifact_path``, ``artifact_hash``, and
        ``status=\"completed\"``.

    Raises:
        ValueError: If ``workspace_path`` or ``workflow_id`` is missing.
        FileNotFoundError: If the candidate directory does not exist.
    """
    workspace_path = state.get("workspace_path")
    workflow_id = state.get("workflow_id")
    if not workspace_path or not workflow_id:
        raise ValueError("workspace_path and workflow_id are required for packaging")

    candidate = candidate_dir(workspace_path)
    if not candidate.exists():
        raise FileNotFoundError(f"Candidate directory not found: {candidate}")

    archive_path = final_dir(workspace_path) / f"{workflow_id}.zip"
    artifact_hash = create_zip_archive(candidate, archive_path)

    return {
        "artifact_path": str(archive_path),
        "artifact_hash": artifact_hash,
        "status": "completed",
    }
