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
from codegen_workflow.revision import plan_diff_payload
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
    """Return a fixed validated plan, with revision fields when requested."""
    change = state.get("change_request") or {}
    current = state.get("plan") or {}
    revised = {**SAMPLE_PLAN, "objective": state["user_request"]}
    if current and change:
        feedback = str(change.get("feedback") or "").lower()
        if "auth" in feedback or "jwt" in feedback:
            revised = {
                **revised,
                "file_manifest": [
                    {"path": "src/app.py", "purpose": "app"},
                    {"path": "src/auth.py", "purpose": "auth"},
                    {"path": "tests/test_app.py", "purpose": "tests"},
                    {"path": "tests/test_auth.py", "purpose": "auth tests"},
                    {"path": "README.md", "purpose": "docs"},
                    {"path": "requirements.txt", "purpose": "deps"},
                ],
            }
        elif "payment" in feedback or "remove" in feedback:
            revised = {
                **revised,
                "file_manifest": [
                    {"path": "src/app.py", "purpose": "app"},
                    {"path": "tests/test_app.py", "purpose": "tests"},
                    {"path": "README.md", "purpose": "docs"},
                    {"path": "requirements.txt", "purpose": "deps"},
                ],
            }
        return {
            "previous_plan": current,
            "plan": revised,
            "plan_diff": plan_diff_payload(current, revised),
            "planner_errors": [],
            "change_request": {},
            "status": "coding",
        }
    return {
        "plan": revised,
        "planner_errors": [],
        "previous_plan": {},
        "plan_diff": {},
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
        "status": "reviewing",
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


def _run_to_human_gate(graph, thread_id: str, request: str = "Build a hello CLI"):
    """Invoke until the human gate interrupt (after automated review)."""
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke({"user_request": request}, config=config)
    return result, config


def test_plain_text_workflow_startup(tmp_path: Path) -> None:
    """Workflow accepts only user_request and starts successfully."""
    graph = _graph(tmp_path)
    result, config = _run_to_human_gate(graph, "startup-1")
    assert "__interrupt__" in result
    state = graph.get_state(config).values
    assert state["user_request"] == "Build a hello CLI"
    assert state["workflow_id"] == "startup-1"
    assert state["status"] == "awaiting_reviewer_approval"


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
    _run_to_human_gate(graph, "planner-coder-1")
    assert coder.call_count == 1


def test_coder_to_verification_routing(tmp_path: Path) -> None:
    """Coder always flows into verification."""
    verify = MagicMock(side_effect=_mock_verify)
    graph = _graph(tmp_path, verify=verify)
    _run_to_human_gate(graph, "coder-verify-1")
    assert verify.call_count == 1


def test_verification_to_human_gate_routing(tmp_path: Path) -> None:
    """Verification and automated review precede the sole human gate."""
    reviewer = MagicMock(side_effect=_mock_reviewer)
    graph = _graph(tmp_path, reviewer=reviewer)
    result, _ = _run_to_human_gate(graph, "verify-gate-1")
    interrupt = result["__interrupt__"][0]
    payload = interrupt.value
    assert reviewer.call_count == 1
    assert payload["gate"] == "reviewer"
    assert "verification_report" in payload
    assert "review_report" in payload
    assert "generated_file_tree" in payload


def test_verify_routes_to_reviewer_before_human_gate(tmp_path: Path) -> None:
    """Automated reviewer runs before any human interrupt."""
    reviewer = MagicMock(side_effect=_mock_reviewer)
    graph = _graph(tmp_path, reviewer=reviewer)
    result, _ = _run_to_human_gate(graph, "auto-review-1")
    assert reviewer.call_count == 1
    assert result["__interrupt__"][0].value["gate"] == "reviewer"


def test_human_change_request_to_planner_then_coder(tmp_path: Path) -> None:
    """request_changes routes through planner revision then coder."""
    planner = MagicMock(side_effect=_mock_planner)
    coder = MagicMock(side_effect=_mock_coder)
    graph = _graph(tmp_path, planner=planner, coder=coder)
    _, config = _run_to_human_gate(graph, "changes-coder-1")
    assert planner.call_count == 1
    assert coder.call_count == 1
    graph.invoke(
        Command(resume={"decision": "request_changes", "feedback": "fix tests"}),
        config=config,
    )
    assert planner.call_count == 2
    assert coder.call_count == 2
    state = graph.get_state(config).values
    assert state.get("previous_plan")
    assert "plan_diff" in state


def test_human_replan_to_planner(tmp_path: Path) -> None:
    """replan at the human gate returns to the planner."""
    planner = MagicMock(side_effect=_mock_planner)
    graph = _graph(tmp_path, planner=planner)
    _, config = _run_to_human_gate(graph, "replan-planner-1")
    assert planner.call_count == 1
    graph.invoke(
        Command(resume={"decision": "replan", "feedback": "switch to FastAPI"}),
        config=config,
    )
    assert planner.call_count == 2
    state = graph.get_state(config).values
    assert state.get("previous_plan")
    assert state.get("plan_diff") is not None


def test_reviewer_approval_to_packaging(tmp_path: Path) -> None:
    """Human approve packages a ZIP artifact for download."""
    graph = _graph(tmp_path)
    _, config = _run_to_human_gate(graph, "package-1")
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


def test_reviewer_change_request_to_planner_then_coder(tmp_path: Path) -> None:
    """request_changes at the human gate revises via planner then coder."""
    planner = MagicMock(side_effect=_mock_planner)
    coder = MagicMock(side_effect=_mock_coder)
    graph = _graph(tmp_path, planner=planner, coder=coder)
    _, config = _run_to_human_gate(graph, "review-changes-1")
    assert planner.call_count == 1
    assert coder.call_count == 1
    graph.invoke(
        Command(resume={"decision": "request_changes", "feedback": "fix bug"}),
        config=config,
    )
    assert planner.call_count == 2
    assert coder.call_count == 2


def test_add_feature_request_changes_flow(tmp_path: Path) -> None:
    """request_changes can add authentication files through planner revision."""
    initial_plan = {
        **SAMPLE_PLAN,
        "file_manifest": [
            {"path": "src/app.py", "purpose": "app"},
            {"path": "tests/test_app.py", "purpose": "tests"},
            {"path": "README.md", "purpose": "docs"},
            {"path": "requirements.txt", "purpose": "deps"},
        ],
    }

    def planner(state: dict[str, Any]) -> dict[str, Any]:
        change = state.get("change_request") or {}
        current = state.get("plan") or {}
        if current and change:
            revised = {
                **initial_plan,
                "file_manifest": [
                    {"path": "src/app.py", "purpose": "app"},
                    {"path": "src/auth.py", "purpose": "auth"},
                    {"path": "tests/test_app.py", "purpose": "tests"},
                    {"path": "tests/test_auth.py", "purpose": "auth tests"},
                    {"path": "README.md", "purpose": "docs"},
                    {"path": "requirements.txt", "purpose": "deps"},
                ],
            }
            return {
                "previous_plan": current,
                "plan": revised,
                "plan_diff": plan_diff_payload(current, revised),
                "planner_errors": [],
                "change_request": {},
                "status": "coding",
            }
        return {
            "plan": initial_plan,
            "planner_errors": [],
            "previous_plan": {},
            "plan_diff": {},
            "status": "coding",
        }

    def coder(state: dict[str, Any]) -> dict[str, Any]:
        workspace = Path(state["workspace_path"])
        candidate = workspace / "candidate"
        candidate.mkdir(parents=True, exist_ok=True)
        created: list[str] = []
        for entry in (state.get("plan") or {}).get("file_manifest") or []:
            rel = str(entry["path"])
            path = candidate / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_text(f"# {rel}\n", encoding="utf-8")
                created.append(rel)
            else:
                path.write_text(path.read_text(encoding="utf-8") + "\n# touched\n", encoding="utf-8")
        iteration = int(state.get("iteration") or 0) + 1
        files = [p.relative_to(candidate).as_posix() for p in candidate.rglob("*") if p.is_file()]
        return {
            "generated_files": files,
            "file_hashes": {path: "hash" for path in files},
            "coder_result": {
                "summary": "reconciled plan",
                "created_files": created,
                "modified_files": [],
                "deleted_files": [],
                "unresolved_issues": [],
                "feedback_resolutions": {},
                "manifest_compliance": {path: True for path in files},
            },
            "iteration": iteration,
            "status": "verifying",
        }

    graph = _graph(tmp_path, planner=planner, coder=coder)
    _, config = _run_to_human_gate(graph, "add-auth-1", "Build a small app")
    graph.invoke(
        Command(
            resume={
                "decision": "request_changes",
                "feedback": "Add JWT authentication",
            }
        ),
        config=config,
    )
    state = graph.get_state(config).values
    assert "src/auth.py" in state["plan_diff"]["added"]
    assert "tests/test_auth.py" in state["plan_diff"]["added"]
    assert (Path(state["workspace_path"]) / "candidate" / "src" / "auth.py").is_file()
    assert (Path(state["workspace_path"]) / "candidate" / "tests" / "test_auth.py").is_file()


def test_remove_feature_request_changes_flow(tmp_path: Path) -> None:
    """request_changes can remove payment files through planner revision."""
    initial_plan = {
        **SAMPLE_PLAN,
        "file_manifest": [
            {"path": "src/app.py", "purpose": "app"},
            {"path": "src/payments.py", "purpose": "payments"},
            {"path": "tests/test_app.py", "purpose": "tests"},
            {"path": "tests/test_payments.py", "purpose": "payment tests"},
            {"path": "README.md", "purpose": "docs"},
            {"path": "requirements.txt", "purpose": "deps"},
        ],
    }

    def planner(state: dict[str, Any]) -> dict[str, Any]:
        change = state.get("change_request") or {}
        current = state.get("plan") or {}
        if current and change:
            revised = {
                **initial_plan,
                "file_manifest": [
                    {"path": "src/app.py", "purpose": "app"},
                    {"path": "tests/test_app.py", "purpose": "tests"},
                    {"path": "README.md", "purpose": "docs"},
                    {"path": "requirements.txt", "purpose": "deps"},
                ],
            }
            return {
                "previous_plan": current,
                "plan": revised,
                "plan_diff": plan_diff_payload(current, revised),
                "planner_errors": [],
                "change_request": {},
                "status": "coding",
            }
        return {
            "plan": initial_plan,
            "planner_errors": [],
            "previous_plan": {},
            "plan_diff": {},
            "status": "coding",
        }

    def coder(state: dict[str, Any]) -> dict[str, Any]:
        workspace = Path(state["workspace_path"])
        candidate = workspace / "candidate"
        candidate.mkdir(parents=True, exist_ok=True)
        deleted: list[str] = []
        for entry in (state.get("plan") or {}).get("file_manifest") or []:
            rel = str(entry["path"])
            path = candidate / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_text(f"# {rel}\n", encoding="utf-8")
        for rel in (state.get("plan_diff") or {}).get("removed") or []:
            path = candidate / rel
            if path.exists():
                path.unlink()
                deleted.append(rel)
        iteration = int(state.get("iteration") or 0) + 1
        files = [p.relative_to(candidate).as_posix() for p in candidate.rglob("*") if p.is_file()]
        return {
            "generated_files": files,
            "file_hashes": {path: "hash" for path in files},
            "coder_result": {
                "summary": "removed payments",
                "created_files": [],
                "modified_files": [],
                "deleted_files": deleted,
                "unresolved_issues": [],
                "feedback_resolutions": {},
                "manifest_compliance": {path: True for path in files},
            },
            "iteration": iteration,
            "status": "verifying",
        }

    graph = _graph(tmp_path, planner=planner, coder=coder)
    _, config = _run_to_human_gate(graph, "remove-pay-1", "Build payments app")

    # Seed payment files that the revision must delete.
    first = graph.get_state(config).values
    candidate = Path(first["workspace_path"]) / "candidate"
    (candidate / "src").mkdir(parents=True, exist_ok=True)
    (candidate / "tests").mkdir(parents=True, exist_ok=True)
    (candidate / "src" / "payments.py").write_text("pay\n", encoding="utf-8")
    (candidate / "tests" / "test_payments.py").write_text("test\n", encoding="utf-8")

    graph.invoke(
        Command(
            resume={
                "decision": "request_changes",
                "feedback": "Remove the payments feature",
            }
        ),
        config=config,
    )
    state = graph.get_state(config).values
    assert "src/payments.py" in state["plan_diff"]["removed"]
    assert "tests/test_payments.py" in state["plan_diff"]["removed"]
    assert "src/payments.py" in state["coder_result"]["deleted_files"]
    assert not (candidate / "src" / "payments.py").exists()
    assert not (candidate / "tests" / "test_payments.py").exists()


def test_reviewer_replan_to_planner(tmp_path: Path) -> None:
    """replan at the human gate returns to the planner."""
    planner = MagicMock(side_effect=_mock_planner)
    graph = _graph(tmp_path, planner=planner)
    _, config = _run_to_human_gate(graph, "review-replan-1")
    assert planner.call_count == 1
    graph.invoke(
        Command(resume={"decision": "replan", "feedback": "new architecture"}),
        config=config,
    )
    assert planner.call_count == 2


def test_abort_routing(tmp_path: Path) -> None:
    """Abort at the human gate ends without packaging."""
    graph = _graph(tmp_path)
    _, config = _run_to_human_gate(graph, "abort-1")
    final = graph.invoke(
        Command(resume={"decision": "abort", "feedback": "stop"}),
        config=config,
    )
    assert final.get("status") == "aborted"
    assert final.get("errors")
    assert final["errors"][-1]["type"] == "aborted"
    assert not final.get("artifact_path")


def test_maximum_iteration_enforcement(tmp_path: Path) -> None:
    """Looping request_changes ends with max_iterations_reached."""
    graph = _graph(tmp_path)
    _, config = _run_to_human_gate(graph, "max-iter-1")

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


def test_invalid_human_decision_rejected(tmp_path: Path) -> None:
    """Unsupported resume payloads are rejected by schema validation."""
    graph = _graph(tmp_path)
    _, config = _run_to_human_gate(graph, "invalid-decision-1")
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
    final = graph.invoke(
        Command(resume={"decision": "approve", "feedback": "looks good"}),
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
    assert ("verify", "reviewer") in edge_pairs
    assert ("verify", "coder_human_gate") not in edge_pairs
    assert ("coder_human_gate", "reviewer") not in edge_pairs


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
