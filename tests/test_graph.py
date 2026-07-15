"""End-to-end and structural tests for the workflow graph."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from codegen_workflow.graph import build_graph, create_workflow, run_config_for_thread
from codegen_workflow.routing import MAX_ITERATIONS, STATUS_INVALID_INPUT, STATUS_MAX_ITERATIONS
from codegen_workflow.workspace import create_workflow_id, initialize_workspace_node


SAMPLE_PLAN = {
    "project_name": "hello_cli",
    "objective": "A hello world CLI",
    "language": "python",
    "framework": None,
    "architecture_pattern": "simple-cli",
    "dependencies": [],
    "epics": [],
    "stories": [],
    "tasks": [],
    "file_manifest": [{"path": "hello.py", "purpose": "entry"}],
    "install_commands": [],
    "validation_commands": ["python3", "-c", "print('ok')"],
    "run_command": "python3 hello.py",
    "risks": [],
}


def _mock_planner(state: dict[str, Any]) -> dict[str, Any]:
    """Return a fixed validated plan."""
    return {
        "plan": {**SAMPLE_PLAN, "objective": state["user_request"]},
        "planner_errors": [],
        "status": "coding",
    }


def _mock_coder(state: dict[str, Any]) -> dict[str, Any]:
    """Materialize a tiny candidate project and bump iteration."""
    workspace = Path(state["workspace_path"])
    candidate = workspace / "candidate"
    candidate.mkdir(parents=True, exist_ok=True)
    (candidate / "hello.py").write_text("print('hello')\n", encoding="utf-8")
    (candidate / "README.md").write_text("# hello\n", encoding="utf-8")
    iteration = int(state.get("iteration") or 0) + 1
    return {
        "generated_files": ["hello.py", "README.md"],
        "file_hashes": {"hello.py": "hash1", "README.md": "hash2"},
        "coder_result": {
            "summary": f"Generated hello CLI (iteration {iteration})",
            "created_files": ["hello.py", "README.md"],
            "modified_files": [],
            "deleted_files": [],
            "unresolved_issues": [],
            "feedback_resolutions": {},
            "manifest_compliance": {"hello.py": True},
        },
        "iteration": iteration,
        "status": "verifying",
    }


def _mock_verify(state: dict[str, Any]) -> dict[str, Any]:
    """Return a deterministic passing verification report."""
    return {
        "verification_report": {
            "passed": True,
            "overall_status": "passed",
            "commands": [
                {
                    "name": "validate_0",
                    "command": ["python3", "-c", "print('ok')"],
                    "exit_code": 0,
                    "stdout": "ok\n",
                    "stderr": "",
                    "duration_seconds": 0.01,
                    "skipped": False,
                }
            ],
            "errors": [],
        },
        "status": "awaiting_coder_approval",
    }


def _mock_reviewer(state: dict[str, Any]) -> dict[str, Any]:
    """Return a structured approving review."""
    return {
        "review_report": {
            "verdict": "approve",
            "acceptance_criteria_results": {"runs": True},
            "manifest_results": {"hello.py": True},
            "reviewed_files": list(state.get("generated_files") or []),
            "findings": [],
            "residual_risks": [],
            "summary": "Approved",
        },
        "status": "awaiting_reviewer_approval",
    }


def _graph(tmp_path: Path, **overrides: Any):
    """Build a test graph with mocked agents."""
    return build_graph(
        checkpointer=InMemorySaver(),
        planner=overrides.get("planner", _mock_planner),
        coder=overrides.get("coder", _mock_coder),
        reviewer=overrides.get("reviewer", _mock_reviewer),
        verify=overrides.get("verify", _mock_verify),
        workspace_base_dir=tmp_path,
    )


def _run_to_coder_gate(graph, thread_id: str, request: str = "Build a hello CLI"):
    """Invoke until the first coder human-gate interrupt."""
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke({"user_request": request}, config=config)
    return result, config


def test_plain_text_workflow_startup(tmp_path: Path) -> None:
    """Workflow accepts only user_request and starts successfully."""
    graph = _graph(tmp_path)
    result, config = _run_to_coder_gate(graph, "startup-1")
    assert "__interrupt__" in result
    state = graph.get_state(config).values
    assert state["user_request"] == "Build a hello CLI"
    assert state["workflow_id"] == "startup-1"
    assert state["status"] == "awaiting_coder_approval"


def test_empty_request_rejected_by_graph(tmp_path: Path) -> None:
    """Empty requests end with an explicit invalid_input status."""
    graph = _graph(tmp_path)
    config = run_config_for_thread("empty-request-1")
    result = graph.invoke({"user_request": "  "}, config=config)
    assert result["status"] == STATUS_INVALID_INPUT
    assert result["errors"]
    assert "__interrupt__" not in result


def test_uuid_generation(tmp_path: Path) -> None:
    """Initialization assigns a valid workflow UUID when no thread id is given."""
    update = initialize_workspace_node(
        {"user_request": "demo"},
        base_dir=tmp_path,
    )
    uuid.UUID(update["workflow_id"])
    uuid.UUID(create_workflow_id())


def test_workspace_creation(tmp_path: Path) -> None:
    """Initialization creates the isolated workspace layout."""
    update = initialize_workspace_node(
        {"user_request": "demo"},
        base_dir=tmp_path,
    )
    root = Path(update["workspace_path"])
    assert root.is_dir()
    for name in ("candidate", "snapshots", "reports", "final"):
        assert (root / name).is_dir()
    assert update["iteration"] == 0
    assert update["max_iterations"] == MAX_ITERATIONS
    assert update["feedback_history"] == []


def test_thread_id_aligned_with_workflow_id(tmp_path: Path) -> None:
    """Runnable thread_id is reused as workflow_id for checkpoint alignment."""
    graph = _graph(tmp_path)
    thread_id = str(uuid.uuid4())
    config = run_config_for_thread(thread_id)
    graph.invoke({"user_request": "Build a hello CLI"}, config=config)
    state = graph.get_state(config).values
    assert state["workflow_id"] == thread_id
    assert thread_id in state["workspace_path"]


def test_planner_to_coder_routing(tmp_path: Path) -> None:
    """After planning, the coder node executes (never skipped)."""
    coder = MagicMock(side_effect=_mock_coder)
    graph = _graph(tmp_path, coder=coder)
    _run_to_coder_gate(graph, "planner-coder-1")
    assert coder.call_count == 1


def test_coder_to_verification_routing(tmp_path: Path) -> None:
    """Coder always flows into verification."""
    verify = MagicMock(side_effect=_mock_verify)
    graph = _graph(tmp_path, verify=verify)
    _run_to_coder_gate(graph, "coder-verify-1")
    assert verify.call_count == 1


def test_verification_to_human_gate_routing(tmp_path: Path) -> None:
    """Verification is followed by an interrupt at the coder human gate."""
    graph = _graph(tmp_path)
    result, _ = _run_to_coder_gate(graph, "verify-gate-1")
    interrupt = result["__interrupt__"][0]
    payload = interrupt.value
    assert payload["gate"] == "coder"
    assert "verification_report" in payload
    assert "generated_file_tree" in payload


def test_coder_approval_to_reviewer(tmp_path: Path) -> None:
    """Approving at the coder gate invokes the reviewer."""
    reviewer = MagicMock(side_effect=_mock_reviewer)
    graph = _graph(tmp_path, reviewer=reviewer)
    _, config = _run_to_coder_gate(graph, "approve-reviewer-1")
    result = graph.invoke(
        Command(resume={"decision": "approve", "feedback": "ok"}),
        config=config,
    )
    assert reviewer.call_count == 1
    assert "__interrupt__" in result
    assert result["__interrupt__"][0].value["gate"] == "reviewer"


def test_coder_change_request_to_coder(tmp_path: Path) -> None:
    """request_changes at the coder gate re-invokes the coder."""
    coder = MagicMock(side_effect=_mock_coder)
    graph = _graph(tmp_path, coder=coder)
    _, config = _run_to_coder_gate(graph, "changes-coder-1")
    assert coder.call_count == 1
    graph.invoke(
        Command(resume={"decision": "request_changes", "feedback": "fix tests"}),
        config=config,
    )
    assert coder.call_count == 2


def test_coder_replan_to_planner(tmp_path: Path) -> None:
    """replan at the coder gate returns to the planner."""
    planner = MagicMock(side_effect=_mock_planner)
    graph = _graph(tmp_path, planner=planner)
    _, config = _run_to_coder_gate(graph, "replan-planner-1")
    assert planner.call_count == 1
    graph.invoke(
        Command(resume={"decision": "replan", "feedback": "switch to FastAPI"}),
        config=config,
    )
    assert planner.call_count == 2


def test_reviewer_approval_to_packaging(tmp_path: Path) -> None:
    """Final human approval packages a ZIP artifact."""
    graph = _graph(tmp_path)
    _, config = _run_to_coder_gate(graph, "package-1")
    graph.invoke(
        Command(resume={"decision": "approve", "feedback": ""}),
        config=config,
    )
    final = graph.invoke(
        Command(resume={"decision": "approve", "feedback": ""}),
        config=config,
    )
    assert final["status"] == "completed"
    assert final["artifact_path"]
    assert final["artifact_hash"]
    assert Path(final["artifact_path"]).is_file()
    assert final["generated_files"]
    assert final["verification_report"]
    assert final["review_report"]


def test_reviewer_change_request_to_coder(tmp_path: Path) -> None:
    """request_changes at the reviewer gate returns to the coder."""
    coder = MagicMock(side_effect=_mock_coder)
    graph = _graph(tmp_path, coder=coder)
    _, config = _run_to_coder_gate(graph, "review-changes-1")
    graph.invoke(
        Command(resume={"decision": "approve", "feedback": ""}),
        config=config,
    )
    assert coder.call_count == 1
    graph.invoke(
        Command(resume={"decision": "request_changes", "feedback": "fix bug"}),
        config=config,
    )
    assert coder.call_count == 2


def test_reviewer_replan_to_planner(tmp_path: Path) -> None:
    """replan at the reviewer gate returns to the planner."""
    planner = MagicMock(side_effect=_mock_planner)
    graph = _graph(tmp_path, planner=planner)
    _, config = _run_to_coder_gate(graph, "review-replan-1")
    graph.invoke(
        Command(resume={"decision": "approve", "feedback": ""}),
        config=config,
    )
    assert planner.call_count == 1
    graph.invoke(
        Command(resume={"decision": "replan", "feedback": "new architecture"}),
        config=config,
    )
    assert planner.call_count == 2


def test_abort_routing(tmp_path: Path) -> None:
    """Abort at the coder gate ends without packaging."""
    graph = _graph(tmp_path)
    _, config = _run_to_coder_gate(graph, "abort-1")
    final = graph.invoke(
        Command(resume={"decision": "abort", "feedback": "stop"}),
        config=config,
    )
    assert final.get("status") == "aborted"
    assert final.get("errors")
    assert final["errors"][-1]["type"] == "aborted"
    assert not final.get("artifact_path")


def test_abort_routing_from_reviewer_gate(tmp_path: Path) -> None:
    """Abort at the reviewer gate ends without packaging."""
    graph = _graph(tmp_path)
    _, config = _run_to_coder_gate(graph, "abort-reviewer-1")
    graph.invoke(
        Command(resume={"decision": "approve", "feedback": ""}),
        config=config,
    )
    final = graph.invoke(
        Command(resume={"decision": "abort", "feedback": "reject"}),
        config=config,
    )
    assert final.get("status") == "aborted"
    assert not final.get("artifact_path")


def test_maximum_iteration_enforcement(tmp_path: Path) -> None:
    """Looping request_changes ends with max_iterations_reached."""
    graph = _graph(tmp_path)
    _, config = _run_to_coder_gate(graph, "max-iter-1")

    for index in range(MAX_ITERATIONS - 1):
        graph.invoke(
            Command(
                resume={
                    "decision": "request_changes",
                    "feedback": f"change {index}",
                }
            ),
            config=config,
        )

    state = graph.get_state(config).values
    assert state["iteration"] == MAX_ITERATIONS

    final = graph.invoke(
        Command(resume={"decision": "request_changes", "feedback": "one more"}),
        config=config,
    )
    assert final.get("status") == STATUS_MAX_ITERATIONS
    assert final.get("errors")
    assert final["errors"][-1]["type"] == STATUS_MAX_ITERATIONS
    assert not final.get("artifact_path")
    assert Path(final["workspace_path"]).exists()
    assert final["generated_files"]


def test_maximum_iteration_enforcement_from_reviewer_gate(tmp_path: Path) -> None:
    """Reviewer request_changes also respects the iteration budget."""
    graph = _graph(tmp_path)
    _, config = _run_to_coder_gate(graph, "max-iter-reviewer-1")

    # Exhaust iterations via coder request_changes, then approve into reviewer.
    for index in range(MAX_ITERATIONS - 1):
        graph.invoke(
            Command(
                resume={
                    "decision": "request_changes",
                    "feedback": f"coder change {index}",
                }
            ),
            config=config,
        )

    graph.invoke(
        Command(resume={"decision": "approve", "feedback": ""}),
        config=config,
    )
    final = graph.invoke(
        Command(resume={"decision": "request_changes", "feedback": "reviewer more"}),
        config=config,
    )
    assert final.get("status") == STATUS_MAX_ITERATIONS
    assert not final.get("artifact_path")


def test_invalid_human_decision_rejected(tmp_path: Path) -> None:
    """Unsupported resume payloads are rejected by schema validation."""
    graph = _graph(tmp_path)
    _, config = _run_to_coder_gate(graph, "invalid-decision-1")
    with pytest.raises(ValueError, match="Unsupported human decision"):
        graph.invoke(
            Command(resume={"decision": "ship_it", "feedback": "nope"}),
            config=config,
        )


def test_successful_end_to_end_with_mocked_agents(tmp_path: Path) -> None:
    """Happy path from plain-text request to completed ZIP artifact."""
    graph = create_workflow(
        checkpointer=InMemorySaver(),
        workspace_base_dir=tmp_path,
        planner=_mock_planner,
        coder=_mock_coder,
        reviewer=_mock_reviewer,
        verify=_mock_verify,
    )
    config = run_config_for_thread()
    graph.invoke({"user_request": "Build a hello CLI"}, config=config)
    graph.invoke(
        Command(resume={"decision": "approve", "feedback": "coder ok"}),
        config=config,
    )
    final = graph.invoke(
        Command(resume={"decision": "approve", "feedback": "reviewer ok"}),
        config=config,
    )

    assert final["status"] == "completed"
    assert final["workflow_id"] == config["configurable"]["thread_id"]
    assert final["artifact_path"].endswith(".zip")
    assert len(final["artifact_hash"]) == 64
    assert "hello.py" in final["generated_files"]
    assert final["verification_report"]["passed"] is True
    assert final["review_report"]["verdict"] == "approve"


def test_coder_cannot_bypass_verification(tmp_path: Path) -> None:
    """Compiled graph has an edge from coder to verify only."""
    graph = _graph(tmp_path)
    drawable = graph.get_graph()
    edge_pairs = {(edge.source, edge.target) for edge in drawable.edges}
    assert ("coder", "verify") in edge_pairs
    assert ("coder", "reviewer") not in edge_pairs
    assert ("coder", "coder_human_gate") not in edge_pairs
    assert ("verify", "coder_human_gate") in edge_pairs


def test_reviewer_cannot_bypass_human_gate(tmp_path: Path) -> None:
    """Reviewer always flows into the reviewer human gate."""
    graph = _graph(tmp_path)
    drawable = graph.get_graph()
    edge_pairs = {(edge.source, edge.target) for edge in drawable.edges}
    assert ("reviewer", "reviewer_human_gate") in edge_pairs
    assert ("reviewer", "package_project") not in edge_pairs


def test_packaging_not_reachable_before_final_approval(tmp_path: Path) -> None:
    """Package node is only reachable from the reviewer human gate."""
    graph = _graph(tmp_path)
    drawable = graph.get_graph()
    incoming = {edge.source for edge in drawable.edges if edge.target == "package_project"}
    assert incoming == {"reviewer_human_gate"}
