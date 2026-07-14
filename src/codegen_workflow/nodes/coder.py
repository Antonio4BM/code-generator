"""Coder agent node for materializing a project plan as files.

Writes generated source, configuration, tests, and documentation into
the isolated ``candidate/`` workspace. Full tool-loop generation is owned
by the coder skill; the graph imports this node interface.
"""

from __future__ import annotations

from typing import Any

from codegen_workflow.state import WorkflowState


def coder_node(state: WorkflowState) -> dict[str, Any]:
    """Create or revise project files from the validated plan.

    Reads plan, workspace path, verification/review reports, and feedback
    history. Returns paths and hashes rather than full file contents.

    Args:
        state: Current workflow state after planning or a change request.

    Returns:
        State update with ``generated_files``, ``file_hashes``,
        ``coder_result``, incremented ``iteration``, and ``status``.

    Raises:
        ValueError: If required plan or workspace fields are missing.
        NotImplementedError: Until an LLM-backed coder is configured.
            Tests should mock this node instead of calling it live.
    """
    if not state.get("plan"):
        return {
            "generated_files": [],
            "file_hashes": {},
            "coder_result": {"summary": "missing plan", "unresolved_issues": ["plan"]},
            "iteration": int(state.get("iteration") or 0),
            "status": "coder_failed",
            "errors": [
                {
                    "type": "invalid_input",
                    "message": "plan is required before coding",
                }
            ],
        }
    if not state.get("workspace_path"):
        raise ValueError("workspace_path is required for coding")

    raise NotImplementedError(
        "coder_node requires an LLM-backed implementation; mock it in tests"
    )
