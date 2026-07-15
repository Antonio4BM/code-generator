"""Safe, workspace-scoped file tools for the coder agent.

All paths are resolved under a single root (typically ``candidate/``).
The tools refuse absolute paths, directory traversal, symlink escapes,
forbidden secret filenames, and size-limit violations.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool

# Default per-file and aggregate size limits (bytes).
DEFAULT_MAX_FILE_SIZE = 256 * 1024
DEFAULT_MAX_PROJECT_SIZE = 5 * 1024 * 1024
DEFAULT_MAX_GENERATED_FILES = 100

FORBIDDEN_FILENAMES = frozenset(
    {
        ".env",
        "credentials.json",
        "secrets.json",
        "secret.json",
        "id_rsa",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
    }
)

_FORBIDDEN_BASENAME_RE = re.compile(r"^\.env(\..+)?$", re.IGNORECASE)


class WorkspaceSecurityError(ValueError):
    """Raised when a path or write violates workspace security rules."""


class WorkspaceLimitError(ValueError):
    """Raised when a size or file-count limit would be exceeded."""


def _is_forbidden_name(name: str) -> bool:
    """Return whether a basename is forbidden for generated projects."""
    if name in FORBIDDEN_FILENAMES:
        return True
    return bool(_FORBIDDEN_BASENAME_RE.match(name))


class WorkspaceFileTools:
    """Narrow filesystem toolkit scoped to one workspace root.

    Args:
        root: Absolute or relative directory that becomes the only
            allowed read/write root (resolved on construction).
        max_file_size: Maximum UTF-8 byte length of a single write.
        max_project_size: Maximum total size of all files under root.
        max_generated_files: Maximum number of files allowed under root.
    """

    def __init__(
        self,
        root: Path | str,
        *,
        max_file_size: int = DEFAULT_MAX_FILE_SIZE,
        max_project_size: int = DEFAULT_MAX_PROJECT_SIZE,
        max_generated_files: int = DEFAULT_MAX_GENERATED_FILES,
    ) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_file_size = max_file_size
        self.max_project_size = max_project_size
        self.max_generated_files = max_generated_files
        self.created_files: list[str] = []
        self.modified_files: list[str] = []
        self.deleted_files: list[str] = []
        self.feedback_resolutions: dict[str, str] = {}

    def _relative_of(self, absolute: Path) -> str:
        """Return a POSIX path relative to the workspace root."""
        return absolute.relative_to(self.root).as_posix()

    def resolve_path(self, relative_path: str) -> Path:
        """Resolve a relative path and ensure it stays inside the root.

        Args:
            relative_path: Path relative to the workspace root.

        Returns:
            Absolute resolved path inside the root.

        Raises:
            WorkspaceSecurityError: If the path is absolute, escapes the
                root, is a forbidden name, or uses symlink escape.
        """
        if not relative_path or not str(relative_path).strip():
            raise WorkspaceSecurityError("path must be a non-empty relative path")

        raw = str(relative_path).strip().replace("\\", "/")
        candidate = Path(raw)
        if candidate.is_absolute():
            raise WorkspaceSecurityError(f"absolute paths are not allowed: {raw!r}")
        if raw.startswith("~"):
            raise WorkspaceSecurityError(
                f"home-relative paths are not allowed: {raw!r}"
            )

        parts = [part for part in Path(raw).parts if part not in ("", ".")]
        if any(part == ".." for part in parts):
            raise WorkspaceSecurityError(
                f"path traversal ('..') is not allowed: {raw!r}"
            )
        for part in parts:
            if _is_forbidden_name(part):
                raise WorkspaceSecurityError(f"forbidden filename: {part!r}")

        # Resolve against root; follow symlinks and re-check containment.
        target = (self.root / Path(*parts)).resolve()
        root_resolved = self.root.resolve()
        try:
            target.relative_to(root_resolved)
        except ValueError as exc:
            raise WorkspaceSecurityError(
                f"path escapes workspace root: {raw!r}"
            ) from exc
        return target

    def _project_size_bytes(self) -> int:
        """Return the total byte size of all regular files under root."""
        total = 0
        if not self.root.exists():
            return 0
        for path in self.root.rglob("*"):
            if path.is_file() and not path.is_symlink():
                total += path.stat().st_size
            elif path.is_file() and path.is_symlink():
                # Count symlink targets only if they remain inside the root.
                try:
                    resolved = path.resolve()
                    resolved.relative_to(self.root.resolve())
                    if resolved.is_file():
                        total += resolved.stat().st_size
                except (ValueError, OSError):
                    continue
        return total

    def _file_count(self) -> int:
        """Return the number of regular files under the workspace root."""
        if not self.root.exists():
            return 0
        count = 0
        for path in self.root.rglob("*"):
            if path.is_file():
                count += 1
        return count

    def list_files(self) -> list[str]:
        """List all relative file paths under the workspace root."""
        if not self.root.exists():
            return []
        files: list[str] = []
        for path in sorted(self.root.rglob("*")):
            if path.is_file():
                try:
                    files.append(self._relative_of(path.resolve()))
                except ValueError:
                    continue
        return files

    def read_file(self, path: str) -> str:
        """Read a UTF-8 text file from the workspace.

        Args:
            path: Relative path within the workspace.

        Returns:
            File contents as text.

        Raises:
            WorkspaceSecurityError: On unsafe paths.
            FileNotFoundError: When the file does not exist.
        """
        target = self.resolve_path(path)
        if not target.exists() or not target.is_file():
            raise FileNotFoundError(f"file not found: {path}")
        return target.read_text(encoding="utf-8")

    def write_file(self, path: str, content: str) -> str:
        """Create or overwrite a UTF-8 text file in the workspace.

        Args:
            path: Relative path within the workspace.
            content: Full file contents to write.

        Returns:
            Confirmation string including the relative path.

        Raises:
            WorkspaceSecurityError: On unsafe paths or forbidden names.
            WorkspaceLimitError: When size or file-count limits are hit.
        """
        if content is None:
            raise ValueError("content is required")
        encoded = content.encode("utf-8")
        if len(encoded) > self.max_file_size:
            raise WorkspaceLimitError(
                f"file exceeds max size of {self.max_file_size} bytes"
            )

        target = self.resolve_path(path)
        exists = target.exists()
        previous_size = target.stat().st_size if exists and target.is_file() else 0

        projected_size = self._project_size_bytes() - previous_size + len(encoded)
        if projected_size > self.max_project_size:
            raise WorkspaceLimitError(
                f"project would exceed max size of {self.max_project_size} bytes"
            )

        if not exists and self._file_count() >= self.max_generated_files:
            raise WorkspaceLimitError(
                f"project would exceed max file count of {self.max_generated_files}"
            )

        target.parent.mkdir(parents=True, exist_ok=True)
        # Re-validate parent after mkdir in case of symlink tricks.
        parent_resolved = target.parent.resolve()
        try:
            parent_resolved.relative_to(self.root.resolve())
        except ValueError as exc:
            raise WorkspaceSecurityError(
                f"parent path escapes workspace root: {path!r}"
            ) from exc

        target.write_text(content, encoding="utf-8")
        relative = self._relative_of(target.resolve())
        if exists:
            if relative not in self.modified_files:
                self.modified_files.append(relative)
        else:
            if relative not in self.created_files:
                self.created_files.append(relative)
        return f"wrote {relative} ({len(encoded)} bytes)"

    def delete_file(self, path: str) -> str:
        """Delete a file from the workspace.

        Args:
            path: Relative path within the workspace.

        Returns:
            Confirmation string.

        Raises:
            WorkspaceSecurityError: On unsafe paths.
            FileNotFoundError: When the file does not exist.
        """
        target = self.resolve_path(path)
        if not target.exists() or not target.is_file():
            raise FileNotFoundError(f"file not found: {path}")
        relative = self._relative_of(target.resolve())
        target.unlink()
        if relative not in self.deleted_files:
            self.deleted_files.append(relative)
        # Drop from created/modified trackers when deleted in-session.
        self.created_files = [p for p in self.created_files if p != relative]
        self.modified_files = [p for p in self.modified_files if p != relative]
        return f"deleted {relative}"

    def search_files(self, query: str) -> list[str]:
        """Search file contents for a literal query string.

        Args:
            query: Substring to find (case-sensitive).

        Returns:
            Relative paths of matching files.
        """
        if not query:
            return []
        matches: list[str] = []
        for relative in self.list_files():
            try:
                text = self.read_file(relative)
            except (OSError, UnicodeDecodeError, WorkspaceSecurityError):
                continue
            if query in text:
                matches.append(relative)
        return matches

    def get_file_hash(self, path: str) -> str:
        """Return the SHA-256 hex digest of a workspace file.

        Args:
            path: Relative path within the workspace.

        Returns:
            Lowercase hex SHA-256 digest.
        """
        target = self.resolve_path(path)
        if not target.exists() or not target.is_file():
            raise FileNotFoundError(f"file not found: {path}")
        digest = hashlib.sha256()
        with target.open("rb") as handle:
            for chunk in iter(lambda: handle.read(65536), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def resolve_feedback(self, finding_id: str, resolution: str) -> str:
        """Record a proposed resolution for a feedback finding.

        Args:
            finding_id: Stable identifier for the finding.
            resolution: Description of how generated files address it.

        Returns:
            Confirmation string.
        """
        finding_id = (finding_id or "").strip()
        resolution = (resolution or "").strip()
        if not finding_id:
            raise ValueError("finding_id is required")
        if not resolution:
            raise ValueError("resolution is required")
        self.feedback_resolutions[finding_id] = resolution
        return f"recorded resolution for {finding_id}"

    def file_hashes(self) -> dict[str, str]:
        """Return SHA-256 hashes for every file currently in the workspace."""
        return {path: self.get_file_hash(path) for path in self.list_files()}

    def as_langchain_tools(self) -> list[StructuredTool]:
        """Expose workspace operations as LangChain structured tools."""

        return [
            StructuredTool.from_function(
                name="list_files",
                description="List all relative file paths in the project workspace.",
                func=self.list_files,
            ),
            StructuredTool.from_function(
                name="read_file",
                description="Read a UTF-8 text file by relative path.",
                func=self.read_file,
            ),
            StructuredTool.from_function(
                name="write_file",
                description=(
                    "Create or overwrite a UTF-8 text file at a relative path. "
                    "Creates parent directories as needed."
                ),
                func=self.write_file,
            ),
            StructuredTool.from_function(
                name="delete_file",
                description=(
                    "Delete a file by relative path. Only delete when required "
                    "by the plan or explicit feedback."
                ),
                func=self.delete_file,
            ),
            StructuredTool.from_function(
                name="search_files",
                description="Search workspace file contents for a literal substring.",
                func=self.search_files,
            ),
            StructuredTool.from_function(
                name="get_file_hash",
                description="Return the SHA-256 hash of a workspace file.",
                func=self.get_file_hash,
            ),
            StructuredTool.from_function(
                name="resolve_feedback",
                description=(
                    "Record how a specific finding_id was addressed in the "
                    "generated files. Call only after making the corresponding change."
                ),
                func=self.resolve_feedback,
            ),
        ]

    def invoke_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Invoke a named tool by string and return a string result.

        Args:
            name: Tool name.
            arguments: Keyword arguments for the tool.

        Returns:
            Stringified tool result for the model-tool loop.
        """
        mapping = {
            "list_files": lambda: self.list_files(),
            "read_file": lambda: self.read_file(**arguments),
            "write_file": lambda: self.write_file(**arguments),
            "delete_file": lambda: self.delete_file(**arguments),
            "search_files": lambda: self.search_files(**arguments),
            "get_file_hash": lambda: self.get_file_hash(**arguments),
            "resolve_feedback": lambda: self.resolve_feedback(**arguments),
        }
        if name not in mapping:
            raise ValueError(f"unknown tool: {name}")
        result = mapping[name]()
        if isinstance(result, list):
            return "\n".join(result) if result else "(empty)"
        return str(result)
