"""Greenfield code-generation LangGraph workflow package.

This package assembles planner, coder, and reviewer agents with
deterministic verification, human approval gates, workspace management,
and packaging into a bounded, interruptible workflow graph.
"""

from codegen_workflow.graph import build_graph, create_workflow, run_config_for_thread

__all__ = [
    "build_graph",
    "create_workflow",
    "run_config_for_thread",
    "create_app",
]


def __getattr__(name: str):
    """Lazily export the API app factory.

    Args:
        name: Attribute name being accessed.

    Returns:
        The requested attribute.

    Raises:
        AttributeError: If the name is unknown.
    """
    if name == "create_app":
        from codegen_workflow.api.app import create_app

        return create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

