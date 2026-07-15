"""Tests for interrupt persistence and resume with the same thread ID."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from codegen_workflow.graph import build_graph


def _mock_planner(state: dict[str, Any]) -> dict[str, Any]:
    """Return a minimal valid plan."""
    return {
        "plan": {
            "project_name": "demo",
            "objective": state["user_request"],
            "language": "python",
            "install_commands": [],
            "validation_commands": ["python3", "-c", "print('ok')"],
        },
        "planner_errors": [],
        "status": "coding",
    }


def _mock_coder(state: dict[str, Any]) -> dict[str, Any]:
    """Write a tiny candidate file and increment iteration."""
    workspace = Path(state["workspace_path"])
    candidate = workspace / "candidate"
    candidate.mkdir(parents=True, exist_ok=True)
    target = candidate / "hello.py"
    target.write_text("print('hello')\n", encoding="utf-8")
    iteration = int(state.get("iteration") or 0) + 1
    return {
        "generated_files": ["hello.py"],
        "file_hashes": {"hello.py": "abc"},
        "coder_result": {"summary": f"iteration {iteration}"},
        "iteration": iteration,
        "status": "verifying",
    }


def _mock_verify(state: dict[str, Any]) -> dict[str, Any]:
    """Return a passing verification report without running processes."""
    return {
        "verification_report": {
            "passed": True,
            "overall_status": "passed",
            "commands": [],
            "errors": [],
        },
        "status": "awaiting_coder_approval",
    }


def _mock_reviewer(state: dict[str, Any]) -> dict[str, Any]:
    """Return an approving review report."""
    return {
        "review_report": {
            "verdict": "approve",
            "acceptance_criteria_results": {"ac1": True},
            "findings": [],
            "residual_risks": [],
            "summary": "Looks good",
        },
        "status": "awaiting_reviewer_approval",
    }


def _build(tmp_path: Path):
    """Compile a graph with mocked agents and an in-memory checkpointer."""
    checkpointer = InMemorySaver()
    graph = build_graph(
        checkpointer=checkpointer,
        planner=_mock_planner,
        coder=_mock_coder,
        reviewer=_mock_reviewer,
        verify=_mock_verify,
        workspace_base_dir=tmp_path,
    )
    return graph, checkpointer


def test_state_persists_across_interrupt(tmp_path: Path) -> None:
    """Checkpointed state remains available after a coder-gate interrupt."""
    graph, _ = _build(tmp_path)
    config = {"configurable": {"thread_id": "persist-thread-1"}}

    result = graph.invoke({"user_request": "Build a hello CLI"}, config=config)
    assert "__interrupt__" in result

    snapshot = graph.get_state(config)
    values = snapshot.values
    assert values["workflow_id"]
    assert values["workspace_path"]
    assert values["generated_files"] == ["hello.py"]
    assert values["verification_report"]["passed"] is True
    assert Path(values["workspace_path"]).exists()


def test_resume_with_same_thread_id(tmp_path: Path) -> None:
    """Resuming with the same thread_id continues the interrupted workflow."""
    graph, _ = _build(tmp_path)
    thread_id = "resume-thread-1"
    config = {"configurable": {"thread_id": thread_id}}

    first = graph.invoke({"user_request": "Build a hello CLI"}, config=config)
    assert "__interrupt__" in first

    interrupted = graph.get_state(config)
    assert interrupted.values["workflow_id"]
    workflow_id = interrupted.values["workflow_id"]

    second = graph.invoke(
        Command(resume={"decision": "approve", "feedback": "ship it"}),
        config=config,
    )
    assert "__interrupt__" in second

    third = graph.invoke(
        Command(resume={"decision": "approve", "feedback": "final ok"}),
        config=config,
    )
    assert third["status"] == "completed"
    assert third["workflow_id"] == workflow_id
    assert third["artifact_path"]
    assert third["artifact_hash"]


def test_human_feedback_appended_on_resume(tmp_path: Path) -> None:
    """Human gate feedback is recorded in feedback_history after resume."""
    graph, _ = _build(tmp_path)
    config = {"configurable": {"thread_id": "feedback-thread-1"}}

    graph.invoke({"user_request": "Build a hello CLI"}, config=config)
    graph.invoke(
        Command(
            resume={
                "decision": "request_changes",
                "feedback": "add README",
            }
        ),
        config=config,
    )
    state = graph.get_state(config).values
    assert state["feedback_history"]
    assert state["feedback_history"][0]["feedback"] == "add README"
    assert state["feedback_history"][0]["decision"] == "request_changes"
    assert state["feedback_history"][0]["gate"] == "coder"


def test_workflow_id_matches_thread_id(tmp_path: Path) -> None:
    """workflow_id is aligned with the runnable thread_id."""
    graph, _ = _build(tmp_path)
    thread_id = "aligned-thread-uuid"
    config = {"configurable": {"thread_id": thread_id}}
    graph.invoke({"user_request": "Build a hello CLI"}, config=config)
    assert graph.get_state(config).values["workflow_id"] == thread_id
