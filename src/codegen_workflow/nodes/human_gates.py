"""Human-in-the-loop approval gates using LangGraph ``interrupt()``.

Both gates pause the workflow, surface structured review payloads to the
caller, and persist the human decision into workflow state. Checkpointing
makes these interrupts durable and resumable with the same ``thread_id``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from langgraph.types import interrupt

from codegen_workflow.routing import status_for_terminal_gate
from codegen_workflow.schemas.decisions import DecisionLiteral, HumanDecision
from codegen_workflow.state import WorkflowState
from codegen_workflow.workspace import candidate_dir, snapshot_candidate


def _file_tree(workspace_path: str | None) -> list[str]:
    """Build a sorted relative file tree for the candidate project.

    Args:
        workspace_path: Workflow workspace root, if known.

    Returns:
        Relative POSIX paths of files under ``candidate/``.
    """
    if not workspace_path:
        return []
    root = candidate_dir(workspace_path)
    if not root.exists():
        return []
    files: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            files.append(path.relative_to(root).as_posix())
    return files


def _diff_from_previous_snapshot(
    workspace_path: str | None,
    iteration: int,
) -> str | None:
    """Produce a simple file-list diff versus the previous snapshot.

    Args:
        workspace_path: Workflow workspace root.
        iteration: Current coder iteration number.

    Returns:
        Human-readable diff text, or ``None`` when no prior snapshot.
    """
    if not workspace_path or iteration <= 0:
        return None
    previous = Path(workspace_path) / "snapshots" / f"iteration_{iteration - 1}"
    if not previous.exists():
        return None

    current_files = set(_file_tree(workspace_path))
    previous_files = {
        path.relative_to(previous).as_posix()
        for path in previous.rglob("*")
        if path.is_file()
    }
    added = sorted(current_files - previous_files)
    removed = sorted(previous_files - current_files)
    lines = [
        f"Diff versus iteration_{iteration - 1}:",
        f"  added: {len(added)}",
        f"  removed: {len(removed)}",
    ]
    for path in added[:50]:
        lines.append(f"  + {path}")
    for path in removed[:50]:
        lines.append(f"  - {path}")
    return "\n".join(lines)


def _normalize_decision(raw: Any) -> dict[str, Any]:
    """Validate interrupt resume payload into a decision dictionary.

    Args:
        raw: Value returned by ``interrupt()`` after resume.

    Returns:
        Serialized :class:`HumanDecision` dictionary.

    Raises:
        ValueError: If the payload cannot be parsed as a decision.
    """
    if isinstance(raw, HumanDecision):
        return raw.model_dump()
    if isinstance(raw, dict):
        return HumanDecision.model_validate(raw).model_dump()
    if isinstance(raw, str):
        decision = raw.strip().lower()
        allowed: tuple[DecisionLiteral, ...] = (
            "approve",
            "request_changes",
            "replan",
            "abort",
        )
        if decision not in allowed:
            raise ValueError(f"Unsupported human decision payload: {raw!r}")
        return HumanDecision(
            decision=decision,  # type: ignore[arg-type]
            feedback="",
        ).model_dump()
    raise ValueError(f"Unsupported human decision payload: {raw!r}")


def coder_human_gate(state: WorkflowState) -> dict[str, Any]:
    """Pause for human approval after coder verification.

    Surfaces the user request, plan, file tree, coder summary,
    verification report, iteration, and optional snapshot diff. Appends
    the human response to ``feedback_history``.

    Args:
        state: Workflow state after the verification node.

    Returns:
        State update with ``coder_human_decision``, feedback history, and
        an updated status.
    """
    iteration = int(state.get("iteration") or 0)
    workspace_path = state.get("workspace_path")
    if workspace_path:
        snapshot_candidate(workspace_path, iteration)

    payload = {
        "gate": "coder",
        "user_request": state.get("user_request"),
        "plan": state.get("plan") or {},
        "generated_file_tree": _file_tree(workspace_path),
        "coder_summary": (state.get("coder_result") or {}).get("summary"),
        "verification_report": state.get("verification_report") or {},
        "iteration": iteration,
        "diff_from_previous_snapshot": _diff_from_previous_snapshot(
            workspace_path,
            iteration,
        ),
    }
    resume_value = interrupt(payload)
    decision = _normalize_decision(resume_value)

    history = list(state.get("feedback_history") or [])
    history.append(
        {
            "gate": "coder",
            "iteration": iteration,
            "decision": decision["decision"],
            "feedback": decision.get("feedback", ""),
        }
    )

    update: dict[str, Any] = {
        "coder_human_decision": decision,
        "feedback_history": history,
    }
    feedback = (decision.get("feedback") or "").strip()
    if decision["decision"] == "replan" and feedback:
        planner_feedback = list(state.get("planner_feedback") or [])
        planner_feedback.append(feedback)
        update["planner_feedback"] = planner_feedback

    update["status"] = status_for_terminal_gate("coder", {**state, **update})
    return update


def reviewer_human_gate(state: WorkflowState) -> dict[str, Any]:
    """Pause for final human approval after the reviewer node.

    Surfaces the plan, file tree, verification report, reviewer verdict,
    acceptance-criteria results, findings, residual risks, and iteration.

    Args:
        state: Workflow state after the reviewer node.

    Returns:
        State update with ``reviewer_human_decision``, feedback history,
        and an updated status.
    """
    review_report = state.get("review_report") or {}
    iteration = int(state.get("iteration") or 0)
    workspace_path = state.get("workspace_path")

    payload = {
        "gate": "reviewer",
        "plan": state.get("plan") or {},
        "generated_file_tree": _file_tree(workspace_path),
        "verification_report": state.get("verification_report") or {},
        "reviewer_verdict": review_report.get("verdict"),
        "acceptance_criteria_results": review_report.get(
            "acceptance_criteria_results",
            {},
        ),
        "findings": review_report.get("findings", []),
        "residual_risks": review_report.get("residual_risks", []),
        "iteration": iteration,
    }
    resume_value = interrupt(payload)
    decision = _normalize_decision(resume_value)

    history = list(state.get("feedback_history") or [])
    history.append(
        {
            "gate": "reviewer",
            "iteration": iteration,
            "decision": decision["decision"],
            "feedback": decision.get("feedback", ""),
        }
    )

    update: dict[str, Any] = {
        "reviewer_human_decision": decision,
        "feedback_history": history,
    }
    feedback = (decision.get("feedback") or "").strip()
    if decision["decision"] == "replan" and feedback:
        planner_feedback = list(state.get("planner_feedback") or [])
        planner_feedback.append(feedback)
        update["planner_feedback"] = planner_feedback

    update["status"] = status_for_terminal_gate("reviewer", {**state, **update})
    return update
