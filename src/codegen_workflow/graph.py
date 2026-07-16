"""Assemble the greenfield code-generation LangGraph workflow.

The graph receives only ``user_request``, creates a UUID-scoped
workspace, and sequences planner → coder → verify → reviewer →
human gate → packaging with deterministic, bounded routing.

Automated review runs before any human interrupt. Human
``request_changes`` and ``replan`` decisions both return to the
planner in revision mode before coding again. Human ``approve``
packages the candidate into a downloadable ZIP.

Checkpointing is used for durable workflow state, interruption,
recovery, and resumption — not as chat-memory storage.

``thread_id`` alignment
    Prefer using a UUID as the LangGraph runnable ``thread_id``. The
    initialize node reuses that value as ``workflow_id`` so the
    checkpoint thread, workspace directory, and artifact names stay
    aligned. Callers may use :func:`run_config_for_thread` to build the
    config dict.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph._node import StateNode
from langgraph.graph.state import CompiledStateGraph

from codegen_workflow.nodes.coder import coder_node
from codegen_workflow.nodes.human_gates import reviewer_human_gate
from codegen_workflow.nodes.planner import planner_node
from codegen_workflow.nodes.reviewer import reviewer_node
from codegen_workflow.nodes.verification import verification_node
from codegen_workflow.packaging import package_project_node
from codegen_workflow.routing import (
    route_after_initialize,
    route_after_planner,
    route_after_reviewer_gate,
)
from codegen_workflow.state import WorkflowState
from codegen_workflow.workspace import create_workflow_id, initialize_workspace_node


def run_config_for_thread(thread_id: str | None = None) -> dict[str, Any]:
    """Build a runnable config with a UUID ``thread_id``.

    Args:
        thread_id: Optional existing thread ID. When omitted, a new UUID
            is generated and should be reused for all resumes of the run.

    Returns:
        Config dict suitable for ``invoke`` / ``stream``.
    """
    return {"configurable": {"thread_id": thread_id or create_workflow_id()}}


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
        checkpointer = InMemorySaver()

    def _initialize(state: WorkflowState, config: RunnableConfig) -> dict[str, Any]:
        thread_id = (config.get("configurable") or {}).get("thread_id")
        aligned_id = str(thread_id) if thread_id else None
        return initialize_workspace_node(
            state,
            base_dir=workspace_base_dir,
            workflow_id=aligned_id,
        )

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
    graph.add_node(
        "reviewer",
        reviewer if reviewer is not None else reviewer_node,
    )
    graph.add_node("reviewer_human_gate", reviewer_human_gate)
    graph.add_node("package_project", package_project_node)

    graph.add_edge(START, "initialize_workspace")
    graph.add_conditional_edges(
        "initialize_workspace",
        route_after_initialize,
        {
            "planner": "planner",
            "__end__": END,
        },
    )
    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "coder": "coder",
            "__end__": END,
        },
    )
    # Coder never routes directly to packaging: verification, automated
    # review, and the human gate are mandatory.
    graph.add_edge("coder", "verify")
    graph.add_edge("verify", "reviewer")
    graph.add_edge("reviewer", "reviewer_human_gate")
    graph.add_conditional_edges(
        "reviewer_human_gate",
        route_after_reviewer_gate,
        {
            "package_project": "package_project",
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
