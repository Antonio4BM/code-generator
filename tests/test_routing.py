"""Tests for deterministic routing helpers."""

from __future__ import annotations

import copy

import pytest

from codegen_workflow.routing import (
    MAX_ITERATIONS,
    STATUS_MAX_ITERATIONS,
    iteration_limit_reached,
    route_after_coder_gate,
    route_after_initialize,
    route_after_planner,
    route_after_reviewer_gate,
    status_for_terminal_gate,
    validate_max_iterations,
)


def test_max_iterations_constant() -> None:
    """MAX_ITERATIONS must be exactly 4."""
    assert MAX_ITERATIONS == 4


def test_validate_max_iterations_range() -> None:
    """max_iterations must stay within the approved positive range."""
    assert validate_max_iterations(1) == 1
    assert validate_max_iterations(10) == 10
    with pytest.raises(ValueError):
        validate_max_iterations(0)
    with pytest.raises(ValueError):
        validate_max_iterations(11)


def test_route_after_initialize_success() -> None:
    """Valid initialization routes to the planner."""
    assert (
        route_after_initialize(
            {
                "status": "planning",
                "workflow_id": "wf",
                "workspace_path": "/tmp/wf",
            }
        )
        == "planner"
    )


def test_route_after_initialize_invalid_input() -> None:
    """invalid_input terminates before planning."""
    assert (
        route_after_initialize({"status": "invalid_input", "errors": [{"type": "x"}]})
        == "__end__"
    )


def test_planner_routes_to_coder_on_success() -> None:
    """Successful planner output routes to the coder."""
    assert (
        route_after_planner({"plan": {"project_name": "demo"}, "status": "coding"})
        == "coder"
    )


def test_planner_routes_to_end_on_failure() -> None:
    """Planner errors route to a terminal end state."""
    assert (
        route_after_planner(
            {
                "plan": {},
                "planner_errors": [{"type": "invalid"}],
                "status": "planner_failed",
            }
        )
        == "__end__"
    )


def test_coder_approval_routes_to_reviewer() -> None:
    """Coder-gate approve must go to the reviewer."""
    state = {"coder_human_decision": {"decision": "approve"}, "iteration": 1}
    assert route_after_coder_gate(state) == "reviewer"


def test_coder_request_changes_routes_to_planner() -> None:
    """Coder-gate request_changes enters planner revision mode."""
    state = {
        "coder_human_decision": {"decision": "request_changes"},
        "iteration": 1,
        "max_iterations": 4,
    }
    assert route_after_coder_gate(state) == "planner"


def test_coder_replan_routes_to_planner() -> None:
    """Coder-gate replan returns to the planner."""
    state = {
        "coder_human_decision": {"decision": "replan"},
        "iteration": 1,
        "max_iterations": 4,
    }
    assert route_after_coder_gate(state) == "planner"


def test_coder_abort_routes_to_end() -> None:
    """Coder-gate abort terminates the workflow."""
    state = {"coder_human_decision": {"decision": "abort"}, "iteration": 1}
    assert route_after_coder_gate(state) == "__end__"


def test_reviewer_approval_routes_to_packaging() -> None:
    """Reviewer-gate approve routes to packaging."""
    state = {"reviewer_human_decision": {"decision": "approve"}, "iteration": 1}
    assert route_after_reviewer_gate(state) == "package_project"


def test_reviewer_request_changes_routes_to_planner() -> None:
    """Reviewer-gate request_changes enters planner revision mode."""
    state = {
        "reviewer_human_decision": {"decision": "request_changes"},
        "iteration": 1,
        "max_iterations": 4,
    }
    assert route_after_reviewer_gate(state) == "planner"


def test_reviewer_replan_routes_to_planner() -> None:
    """Reviewer-gate replan returns to the planner."""
    state = {
        "reviewer_human_decision": {"decision": "replan"},
        "iteration": 2,
        "max_iterations": 4,
    }
    assert route_after_reviewer_gate(state) == "planner"


def test_reviewer_abort_routes_to_end() -> None:
    """Reviewer-gate abort terminates the workflow."""
    state = {"reviewer_human_decision": {"decision": "abort"}, "iteration": 1}
    assert route_after_reviewer_gate(state) == "__end__"


def test_iteration_limit_blocks_coder_loop() -> None:
    """Loop-back from coder gate ends at the iteration limit."""
    state = {
        "coder_human_decision": {"decision": "request_changes"},
        "iteration": MAX_ITERATIONS,
        "max_iterations": MAX_ITERATIONS,
    }
    assert iteration_limit_reached(state) is True
    assert route_after_coder_gate(state) == "__end__"
    assert status_for_terminal_gate("coder", state) == STATUS_MAX_ITERATIONS


def test_iteration_limit_blocks_reviewer_loop() -> None:
    """Loop-back from reviewer gate ends at the iteration limit."""
    state = {
        "reviewer_human_decision": {"decision": "replan"},
        "iteration": MAX_ITERATIONS,
        "max_iterations": MAX_ITERATIONS,
    }
    assert route_after_reviewer_gate(state) == "__end__"
    assert status_for_terminal_gate("reviewer", state) == STATUS_MAX_ITERATIONS


def test_routing_functions_do_not_mutate_state() -> None:
    """Routing helpers are pure and leave the input mapping unchanged."""
    original = {
        "plan": {"project_name": "demo"},
        "status": "coding",
        "coder_human_decision": {"decision": "approve"},
        "iteration": 1,
        "max_iterations": 4,
        "workflow_id": "wf",
        "workspace_path": "/tmp/wf",
    }
    snapshot = copy.deepcopy(original)
    route_after_planner(original)  # type: ignore[arg-type]
    route_after_coder_gate(original)  # type: ignore[arg-type]
    route_after_initialize(original)  # type: ignore[arg-type]
    assert original == snapshot
