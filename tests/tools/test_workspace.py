"""Tests for safe workspace file tools."""

from __future__ import annotations

from pathlib import Path

import pytest

from codegen_workflow.tools.workspace import (
    WorkspaceFileTools,
    WorkspaceLimitError,
    WorkspaceSecurityError,
)


@pytest.fixture
def tools(tmp_path: Path) -> WorkspaceFileTools:
    """Fresh workspace tools rooted in a temporary directory."""
    root = tmp_path / "candidate"
    root.mkdir()
    return WorkspaceFileTools(
        root, max_file_size=100, max_project_size=200, max_generated_files=5
    )


def test_create_files_from_empty_workspace(tools: WorkspaceFileTools) -> None:
    """Writing new files populates the empty workspace."""
    assert tools.list_files() == []
    tools.write_file("hello.py", "print('hi')\n")
    assert tools.read_file("hello.py") == "print('hi')\n"
    assert tools.list_files() == ["hello.py"]
    assert tools.created_files == ["hello.py"]


def test_create_nested_directories(tools: WorkspaceFileTools) -> None:
    """write_file creates intermediate directories as needed."""
    tools.write_file("src/pkg/main.py", "x = 1\n")
    assert (tools.root / "src" / "pkg" / "main.py").is_file()
    assert "src/pkg/main.py" in tools.list_files()


def test_update_existing_file(tools: WorkspaceFileTools) -> None:
    """Overwriting an existing file tracks it as modified."""
    tools.write_file("app.py", "v1\n")
    tools.created_files.clear()
    tools.write_file("app.py", "v2\n")
    assert tools.read_file("app.py") == "v2\n"
    assert tools.modified_files == ["app.py"]
    assert tools.created_files == []


def test_delete_obsolete_file(tools: WorkspaceFileTools) -> None:
    """delete_file removes a file and records the deletion."""
    tools.write_file("old.py", "gone\n")
    tools.delete_file("old.py")
    assert tools.list_files() == []
    assert tools.deleted_files == ["old.py"]


def test_reject_absolute_paths(tools: WorkspaceFileTools, tmp_path: Path) -> None:
    """Absolute paths are rejected."""
    absolute = str(tmp_path / "elsewhere.py")
    with pytest.raises(WorkspaceSecurityError, match="absolute"):
        tools.write_file(absolute, "nope")


def test_reject_parent_traversal(tools: WorkspaceFileTools) -> None:
    """Paths containing '..' are rejected before resolution."""
    with pytest.raises(WorkspaceSecurityError, match="traversal"):
        tools.write_file("../escape.py", "nope")
    with pytest.raises(WorkspaceSecurityError, match="traversal"):
        tools.read_file("sub/../../escape.py")


def test_reject_symlink_escape(tools: WorkspaceFileTools, tmp_path: Path) -> None:
    """Symlinks that resolve outside the workspace are rejected."""
    outside = tmp_path / "outside.txt"
    outside.write_text("secret\n", encoding="utf-8")
    link = tools.root / "leak"
    link.symlink_to(outside)
    with pytest.raises(WorkspaceSecurityError, match="escapes"):
        tools.read_file("leak")

    outside_dir = tmp_path / "outside_dir"
    outside_dir.mkdir()
    dir_link = tools.root / "outdir"
    dir_link.symlink_to(outside_dir)
    with pytest.raises(WorkspaceSecurityError, match="escapes"):
        tools.write_file("outdir/x.py", "escaped\n")


def test_reject_forbidden_filenames(tools: WorkspaceFileTools) -> None:
    """Secret and credential filenames cannot be written."""
    with pytest.raises(WorkspaceSecurityError, match="forbidden"):
        tools.write_file(".env", "SECRET=1\n")
    with pytest.raises(WorkspaceSecurityError, match="forbidden"):
        tools.write_file("config/credentials.json", '{"k":"v"}')
    with pytest.raises(WorkspaceSecurityError, match="forbidden"):
        tools.write_file(".env.local", "x=1\n")


def test_enforce_file_size_limit(tools: WorkspaceFileTools) -> None:
    """Oversized single-file writes are rejected."""
    with pytest.raises(WorkspaceLimitError, match="max size"):
        tools.write_file("big.py", "x" * 101)


def test_enforce_project_size_limit(tools: WorkspaceFileTools) -> None:
    """Aggregate project size limits are enforced across writes."""
    tools.write_file("a.py", "a" * 100)
    tools.write_file("b.py", "b" * 90)
    with pytest.raises(WorkspaceLimitError, match="project would exceed"):
        tools.write_file("c.py", "c" * 20)


def test_enforce_max_generated_files(tmp_path: Path) -> None:
    """File-count limits reject additional creates."""
    tools = WorkspaceFileTools(
        tmp_path / "candidate",
        max_file_size=1000,
        max_project_size=10_000,
        max_generated_files=2,
    )
    tools.write_file("a.py", "a\n")
    tools.write_file("b.py", "b\n")
    with pytest.raises(WorkspaceLimitError, match="max file count"):
        tools.write_file("c.py", "c\n")


def test_maintain_file_hashes(tools: WorkspaceFileTools) -> None:
    """get_file_hash and file_hashes track content digests."""
    tools.write_file("a.py", "hello\n")
    digest = tools.get_file_hash("a.py")
    assert len(digest) == 64
    assert tools.file_hashes() == {"a.py": digest}
    tools.write_file("a.py", "hello!\n")
    assert tools.get_file_hash("a.py") != digest


def test_search_files(tools: WorkspaceFileTools) -> None:
    """search_files returns paths containing the query substring."""
    tools.write_file("a.py", "alpha\n")
    tools.write_file("b.py", "beta\n")
    assert tools.search_files("alpha") == ["a.py"]
