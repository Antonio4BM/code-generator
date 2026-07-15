"""Read-only workspace tools for the reviewer agent.

Exposes inspection helpers only. Write/delete operations are intentionally
absent so the reviewer cannot modify generated project files.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import StructuredTool

from codegen_workflow.tools.workspace import WorkspaceFileTools, WorkspaceSecurityError


READ_ONLY_TOOL_NAMES = frozenset(
    {
        "list_files",
        "read_file",
        "search_files",
        "get_file_hash",
        "read_verification_report",
    }
)

FORBIDDEN_MUTATION_TOOL_NAMES = frozenset(
    {
        "write_file",
        "delete_file",
        "resolve_feedback",
    }
)


class ReadOnlyWorkspaceTools:
    """Read-only toolkit scoped to a candidate workspace root.

    Args:
        root: Candidate project root for safe path resolution.
        verification_report: Verification report dict surfaced via tool.
    """

    def __init__(
        self,
        root: Any,
        *,
        verification_report: dict[str, Any] | None = None,
    ) -> None:
        # Reuse path safety from WorkspaceFileTools without exposing writers
        # through this facade's LangChain tool list.
        self._fs = WorkspaceFileTools(root)
        self.verification_report = dict(verification_report or {})
        self.reviewed_files: list[str] = []

    @property
    def root(self) -> Any:
        """Absolute resolved candidate root."""
        return self._fs.root

    def list_files(self) -> list[str]:
        """List relative file paths under the candidate root."""
        return self._fs.list_files()

    def read_file(self, path: str) -> str:
        """Read a UTF-8 workspace file and record it as reviewed."""
        content = self._fs.read_file(path)
        normalized = path.replace("\\", "/")
        if normalized not in self.reviewed_files:
            self.reviewed_files.append(normalized)
        return content

    def search_files(self, query: str) -> list[str]:
        """Search file contents for a literal substring."""
        return self._fs.search_files(query)

    def get_file_hash(self, path: str) -> str:
        """Return the SHA-256 digest of a workspace file."""
        return self._fs.get_file_hash(path)

    def read_verification_report(self) -> str:
        """Return the verification report as formatted JSON text."""
        return json.dumps(self.verification_report, indent=2, sort_keys=True)

    def as_langchain_tools(self) -> list[StructuredTool]:
        """Expose only read-only operations as LangChain tools."""
        return [
            StructuredTool.from_function(
                name="list_files",
                description="List all relative file paths in the candidate workspace.",
                func=self.list_files,
            ),
            StructuredTool.from_function(
                name="read_file",
                description="Read a UTF-8 text file by relative path (read-only).",
                func=self.read_file,
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
                name="read_verification_report",
                description="Read the automated verification report for this iteration.",
                func=self.read_verification_report,
            ),
        ]

    def tool_names(self) -> list[str]:
        """Return the exposed tool names in registration order."""
        return [tool.name for tool in self.as_langchain_tools()]

    def invoke_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Invoke a read-only tool by name.

        Args:
            name: Tool name.
            arguments: Keyword arguments for the tool.

        Returns:
            Stringified tool result.

        Raises:
            ValueError: If the tool is unknown or is a mutation tool.
        """
        if name in FORBIDDEN_MUTATION_TOOL_NAMES:
            raise ValueError(f"mutation tool is not available to the reviewer: {name}")
        mapping = {
            "list_files": lambda: self.list_files(),
            "read_file": lambda: self.read_file(**arguments),
            "search_files": lambda: self.search_files(**arguments),
            "get_file_hash": lambda: self.get_file_hash(**arguments),
            "read_verification_report": lambda: self.read_verification_report(),
        }
        if name not in mapping:
            raise ValueError(f"unknown tool: {name}")
        result = mapping[name]()
        if isinstance(result, list):
            return "\n".join(result) if result else "(empty)"
        return str(result)


__all__ = [
    "FORBIDDEN_MUTATION_TOOL_NAMES",
    "READ_ONLY_TOOL_NAMES",
    "ReadOnlyWorkspaceTools",
    "WorkspaceSecurityError",
]
