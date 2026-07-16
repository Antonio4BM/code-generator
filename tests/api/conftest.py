"""Shared fixtures for API tests with a mocked LangGraph workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from codegen_workflow.api.app import create_app
from codegen_workflow.api.config import APISettings
from codegen_workflow.graph import build_graph


SAMPLE_PLAN = {
    "project_name": "hello_cli",
    "objective": "A hello world CLI",
    "language": "python",
    "install_commands": [],
    "validation_commands": ["python3", "-c", "print('ok')"],
}


def mock_planner(state: dict[str, Any]) -> dict[str, Any]:
    """Return a fixed validated plan without calling an LLM."""
    from codegen_workflow.revision import plan_diff_payload

    change = state.get("change_request") or {}
    current = state.get("plan") or {}
    revised = {**SAMPLE_PLAN, "objective": state["user_request"]}
    if current and change:
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


def mock_coder(state: dict[str, Any]) -> dict[str, Any]:
    """Write a tiny candidate project and increment iteration."""
    workspace = Path(state["workspace_path"])
    candidate = workspace / "candidate"
    candidate.mkdir(parents=True, exist_ok=True)
    (candidate / "hello.py").write_text("print('hello')\n", encoding="utf-8")
    (candidate / "README.md").write_text("# hello\n", encoding="utf-8")
    iteration = int(state.get("iteration") or 0) + 1
    return {
        "generated_files": ["hello.py", "README.md"],
        "file_hashes": {"hello.py": "hash1", "README.md": "hash2"},
        "coder_result": {"summary": f"Generated hello CLI (iteration {iteration})"},
        "iteration": iteration,
        "status": "verifying",
    }


def mock_verify(state: dict[str, Any]) -> dict[str, Any]:
    """Return a passing verification report."""
    return {
        "verification_report": {
            "passed": True,
            "overall_status": "passed",
            "commands": [],
            "errors": [],
        },
        "status": "reviewing",
    }


def mock_reviewer(state: dict[str, Any]) -> dict[str, Any]:
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


def failing_planner(state: dict[str, Any]) -> dict[str, Any]:
    """Force a planner_failed terminal status."""
    return {
        "plan": {},
        "planner_errors": [{"type": "planner_failed", "message": "boom"}],
        "status": "planner_failed",
    }


@pytest.fixture
def workspace_root(tmp_path: Path) -> Path:
    """Provide an isolated workspace parent directory."""
    root = tmp_path / "workspaces"
    root.mkdir()
    return root


@pytest.fixture
def artifact_root(tmp_path: Path) -> Path:
    """Provide an isolated artifact parent directory."""
    root = tmp_path / "artifacts"
    root.mkdir()
    return root


@pytest.fixture
def api_settings(workspace_root: Path, artifact_root: Path) -> APISettings:
    """Build non-production settings pointing at temp directories."""
    return APISettings(
        openai_api_key="test-key",
        workspace_base_dir=workspace_root,
        artifact_base_dir=artifact_root,
        log_level="WARNING",
        allowed_origins=["http://localhost:3000"],
        app_env="test",
        workflow_timeout_seconds=30.0,
    )


@pytest.fixture
def mock_graph(workspace_root: Path):
    """Compile a graph with mocked agent nodes."""
    return build_graph(
        checkpointer=InMemorySaver(),
        planner=mock_planner,
        coder=mock_coder,
        reviewer=mock_reviewer,
        verify=mock_verify,
        workspace_base_dir=workspace_root,
    )


@pytest.fixture
def client(api_settings: APISettings, mock_graph) -> TestClient:
    """HTTP client bound to an app using the mocked graph."""
    app = create_app(settings=api_settings, graph=mock_graph)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def planner_fail_client(api_settings: APISettings, workspace_root: Path) -> TestClient:
    """HTTP client whose planner always fails."""
    graph = build_graph(
        checkpointer=InMemorySaver(),
        planner=failing_planner,
        coder=mock_coder,
        reviewer=mock_reviewer,
        verify=mock_verify,
        workspace_base_dir=workspace_root,
    )
    app = create_app(settings=api_settings, graph=graph)
    with TestClient(app) as test_client:
        yield test_client
