"""Assemble the greenfield code-generation LangGraph workflow.

The graph receives only ``user_request``, creates a UUID-scoped
workspace, and sequences planner → coder → verify → human gates →
reviewer → packaging with deterministic, bounded routing.

Checkpointing is used for durable workflow state, interruption,
recovery, and resumption — not as chat-memory storage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph._node import StateNode
from langgraph.graph.state import CompiledStateGraph

from codegen_workflow.nodes.coder import coder_node
from codegen_workflow.nodes.human_gates import coder_human_gate, reviewer_human_gate
from codegen_workflow.nodes.planner import planner_node
from codegen_workflow.nodes.reviewer import reviewer_node
from codegen_workflow.nodes.verification import verification_node
from codegen_workflow.packaging import package_project_node
from codegen_workflow.routing import (
    route_after_coder_gate,
    route_after_planner,
    route_after_reviewer_gate,
)
from codegen_workflow.state import WorkflowState
from codegen_workflow.workspace import initialize_workspace_node


def build_graph(
    *,
    checkpointer: Any | None = None,
    planner: StateNode[WorkflowState, None] | None = None,
    coder: StateNode[WorkflowState, None] | None = None,
    reviewer: StateNode[WorkflowState, None] | None = None,
    verify: StateNode[WorkflowState, None] | None = None,
    workspace_base_dir: Path | str | None = None,
) -> CompiledStateGraph[WorkflowState, None, WorkflowState, WorkflowState]:
    """Build and compile the code-generation workflow graph.

    Args:
        checkpointer: Injectable checkpointer. Defaults to an in-memory
            saver suitable for local development and tests. Pass a
            persistent checkpointer for production.
        planner: Optional override for the planner node callable.
        coder: Optional override for the coder node callable.
        reviewer: Optional override for the reviewer node callable.
        verify: Optional override for the verification node callable.
        workspace_base_dir: Optional parent directory for workspaces.

    Returns:
        A compiled LangGraph ready for ``invoke`` / ``stream`` with a
        ``thread_id`` in the runnable config.
    """
    if checkpointer is None:
        checkpointer = MemorySaver()

    def _initialize(state: WorkflowState) -> dict[str, Any]:
        return initialize_workspace_node(state, base_dir=workspace_base_dir)

    graph = StateGraph(WorkflowState)

    graph.add_node("initialize_workspace", _initialize)
    graph.add_node(
        "planner",
        planner if planner is not None else planner_node,
    )
    graph.add_node(
        "coder",
        coder if coder is not None else coder_node,
    )
    graph.add_node(
        "verify",
        verify if verify is not None else verification_node,
    )
    graph.add_node("coder_human_gate", coder_human_gate)
    graph.add_node(
        "reviewer",
        reviewer if reviewer is not None else reviewer_node,
    )
    graph.add_node("reviewer_human_gate", reviewer_human_gate)
    graph.add_node("package_project", package_project_node)

    graph.add_edge(START, "initialize_workspace")
    graph.add_edge("initialize_workspace", "planner")
    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "coder": "coder",
            "__end__": END,
        },
    )
    # Coder never routes directly to the reviewer: verification and the
    # coder human gate are mandatory.
    graph.add_edge("coder", "verify")
    graph.add_edge("verify", "coder_human_gate")
    graph.add_conditional_edges(
        "coder_human_gate",
        route_after_coder_gate,
        {
            "reviewer": "reviewer",
            "coder": "coder",
            "planner": "planner",
            "__end__": END,
        },
    )
    # Reviewer verdict is advisory; the human gate controls the transition.
    graph.add_edge("reviewer", "reviewer_human_gate")
    graph.add_conditional_edges(
        "reviewer_human_gate",
        route_after_reviewer_gate,
        {
            "package_project": "package_project",
            "coder": "coder",
            "planner": "planner",
            "__end__": END,
        },
    )
    graph.add_edge("package_project", END)

    return graph.compile(checkpointer=checkpointer)


def create_workflow(
    *,
    checkpointer: Any | None = None,
    workspace_base_dir: Path | str | None = None,
    **node_overrides: Any,
) -> CompiledStateGraph[WorkflowState, None, WorkflowState, WorkflowState]:
    """Create a production-ready workflow with injectable persistence.

    Args:
        checkpointer: In-memory or persistent checkpointer instance.
        workspace_base_dir: Optional parent directory for workspaces.
        **node_overrides: Optional ``planner``, ``coder``, ``reviewer``,
            or ``verify`` callables forwarded to :func:`build_graph`.

    Returns:
        Compiled workflow graph.
    """
    return build_graph(
        checkpointer=checkpointer,
        workspace_base_dir=workspace_base_dir,
        **node_overrides,
    )
