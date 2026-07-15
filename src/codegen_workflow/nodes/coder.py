"""Coder agent node for materializing a project plan as files.

Runs a bounded model–tool loop that writes source, configuration, tests,
and documentation into the isolated ``candidate/`` workspace. Returns
paths, hashes, and a ``CoderResult`` summary rather than file contents.
"""

from __future__ import annotations

import json
import logging
import os
from collections import deque
from typing import Any, Protocol

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import ValidationError

from codegen_workflow.routing import MAX_ITERATIONS
from codegen_workflow.schemas.coder import CoderResult
from codegen_workflow.schemas.plan import (
    FileSpecification,
    PlanValidationError,
    ProjectPlan,
    validate_plan,
)
from codegen_workflow.state import WorkflowState
from codegen_workflow.tools.workspace import (
    DEFAULT_MAX_FILE_SIZE,
    DEFAULT_MAX_GENERATED_FILES,
    DEFAULT_MAX_PROJECT_SIZE,
    WorkspaceFileTools,
    WorkspaceLimitError,
    WorkspaceSecurityError,
)
from codegen_workflow.workspace import candidate_dir

logger = logging.getLogger(__name__)

DEFAULT_CODER_MODEL = os.environ.get("CODER_MODEL", "gpt-4o-mini")

MAX_TOOL_CALLS_PER_ITERATION = 80
MAX_MODEL_RETRIES = 3
MAX_CODER_ITERATIONS = MAX_ITERATIONS

SYSTEM_PROMPT = """\
You are the coder for a greenfield code-generation workflow.
There is no existing repository outside the provided workspace.
Use only the provided tools to inspect and modify files.

Rules:
- The planner plan is authoritative. Do not silently redesign architecture.
- All paths are relative to the project root (candidate workspace).
- Generate complete, runnable files. No TODO stubs for required behavior.
- No ellipses standing in for implementation.
- No hardcoded credentials; use environment variables for secrets.
- Never write forbidden secret files (.env, credentials.json, etc.).
- Include meaningful error handling.
- Keep imports and interfaces consistent across files.
- Prefer the language and framework selected in the plan.
- Do not add dependencies absent from the plan unless essential; if you
  must, record the reason via unresolved issues in your final summary.

Process:
1. Inspect the workspace with list_files / read_file.
2. For initial generation, create every file in the plan file_manifest
   in dependency order (depends_on).
3. For revisions, read existing files first and make targeted changes.
4. Address verification failures, reviewer findings, and human feedback.
5. Call resolve_feedback(finding_id, resolution) only after making the
   corresponding file changes.
6. When finished, stop calling tools and reply with a short plain-text
   summary of what you did.
"""


class ToolCallingModel(Protocol):
    """Minimal protocol for a tool-calling chat model."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        """Bind tools for subsequent invoke calls."""

    def invoke(self, input: Any, config: Any | None = None, **kwargs: Any) -> Any:
        """Invoke the model."""


class CoderLimitError(RuntimeError):
    """Raised when a configured coder execution limit is exceeded."""


def create_coder_llm(
    *,
    model_name: str | None = None,
    temperature: float = 0.0,
) -> BaseChatModel:
    """Create the default AzureChatOpenAI model used by the coder.

    Args:
        model_name: Optional Azure deployment name.
        temperature: Sampling temperature (prefer low for codegen).

    Returns:
        A chat model instance that supports ``bind_tools``.
    """
    from codegen_workflow.llm import create_azure_chat_model

    return create_azure_chat_model(
        model_name=model_name or DEFAULT_CODER_MODEL,
        temperature=temperature,
    )


def topological_file_order(manifest: list[FileSpecification]) -> list[str]:
    """Order manifest paths so dependencies come before dependents.

    Args:
        manifest: Planner file specifications.

    Returns:
        Ordered relative paths. Cycles fall back to declaration order
        for remaining nodes.
    """
    paths = [spec.path.replace("\\", "/") for spec in manifest]
    path_set = set(paths)
    dependents: dict[str, list[str]] = {path: [] for path in paths}
    indegree: dict[str, int] = {path: 0 for path in paths}

    for spec in manifest:
        path = spec.path.replace("\\", "/")
        for dep in spec.depends_on:
            dep_norm = dep.replace("\\", "/")
            if dep_norm in path_set:
                dependents[dep_norm].append(path)
                indegree[path] += 1

    queue = deque(sorted(path for path, degree in indegree.items() if degree == 0))
    ordered: list[str] = []
    while queue:
        node = queue.popleft()
        ordered.append(node)
        for child in sorted(dependents[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)

    if len(ordered) < len(paths):
        remaining = [path for path in paths if path not in set(ordered)]
        ordered.extend(remaining)
    return ordered


def extract_findings(state: WorkflowState) -> list[tuple[str, str]]:
    """Collect stable finding ids and messages from workflow state.

    Args:
        state: Current workflow state.

    Returns:
        Ordered ``(finding_id, description)`` pairs.
    """
    findings: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add(finding_id: str, description: str) -> None:
        fid = finding_id.strip()
        if not fid or fid in seen:
            return
        seen.add(fid)
        findings.append((fid, description.strip()))

    for index, item in enumerate(state.get("feedback_history") or []):
        if not isinstance(item, dict):
            continue
        decision = str(item.get("decision") or "")
        feedback = str(item.get("feedback") or "").strip()
        if decision not in {"request_changes", "replan"} and not feedback:
            continue
        if not feedback and decision != "request_changes":
            continue
        gate = str(item.get("gate") or "human")
        iteration = item.get("iteration", index)
        finding_id = str(
            item.get("id") or item.get("finding_id") or f"{gate}_{iteration}_{index}"
        )
        text = feedback or f"{gate} requested changes"
        add(finding_id, text)

    review = state.get("review_report") or {}
    raw_findings = review.get("findings") or []
    if isinstance(raw_findings, list):
        for index, finding in enumerate(raw_findings):
            if isinstance(finding, dict):
                finding_id = str(
                    finding.get("id") or finding.get("finding_id") or f"review_{index}"
                )
                description = str(
                    finding.get("description")
                    or finding.get("message")
                    or finding.get("summary")
                    or finding
                )
                add(finding_id, description)
            else:
                add(f"review_{index}", str(finding))

    verification = state.get("verification_report") or {}
    if verification and verification.get("passed") is False:
        commands = verification.get("commands") or []
        failed = [
            cmd
            for cmd in commands
            if isinstance(cmd, dict) and int(cmd.get("exit_code", 0) or 0) != 0
        ]
        if failed:
            for index, cmd in enumerate(failed):
                name = str(cmd.get("name") or f"cmd_{index}")
                stderr = str(cmd.get("stderr") or cmd.get("stdout") or "failed")
                add(f"verification_{name}", stderr[:500])
        else:
            errors = verification.get("errors") or []
            if errors:
                for index, err in enumerate(errors):
                    add(f"verification_error_{index}", str(err)[:500])
            else:
                add(
                    "verification_failed",
                    str(verification.get("overall_status") or "verification failed"),
                )

    return findings


def build_coder_prompt(
    *,
    user_request: str,
    plan: ProjectPlan,
    iteration: int,
    existing_files: list[str],
    findings: list[tuple[str, str]],
) -> str:
    """Build the human prompt for one coder iteration.

    Args:
        user_request: Original software requirement.
        plan: Validated project plan.
        iteration: Current iteration count before this coder run.
        existing_files: Relative paths already present in the workspace.
        findings: Extracted findings to address.

    Returns:
        Prompt text for the tool-calling model.
    """
    mode = "revision" if existing_files else "initial_generation"
    ordered = topological_file_order(plan.file_manifest)
    findings_block = (
        "\n".join(f"- {fid}: {desc}" for fid, desc in findings) if findings else "None."
    )
    manifest_lines = []
    for spec in plan.file_manifest:
        manifest_lines.append(
            f"- {spec.path} [{spec.file_type}] depends_on={spec.depends_on}: "
            f"{spec.purpose}"
        )

    return "\n\n".join(
        [
            f"## Mode\n{mode} (current iteration={iteration})",
            f"## User request\n{user_request.strip()}",
            "## Plan summary\n"
            f"project_name={plan.project_name}\n"
            f"language={plan.language}\n"
            f"framework={plan.framework}\n"
            f"architecture_pattern={plan.architecture_pattern}\n"
            f"dependencies={plan.dependencies}\n"
            f"install_commands={plan.install_commands}\n"
            f"validation_commands={plan.validation_commands}\n"
            f"run_command={plan.run_command}",
            "## Recommended generation order\n" + "\n".join(f"- {p}" for p in ordered),
            "## File manifest\n" + "\n".join(manifest_lines),
            "## Existing workspace files\n"
            + (
                "\n".join(f"- {p}" for p in existing_files)
                if existing_files
                else "None."
            ),
            "## Findings to address\n" + findings_block,
            "Use tools to implement the required files, then stop with a short summary.",
        ]
    )


def run_tool_loop(
    llm: Any,
    tools: WorkspaceFileTools,
    messages: list[Any],
    *,
    max_tool_calls: int = MAX_TOOL_CALLS_PER_ITERATION,
    max_model_retries: int = MAX_MODEL_RETRIES,
) -> tuple[list[Any], int, str]:
    """Execute a bounded model–tool loop.

    Args:
        llm: Chat model supporting ``bind_tools`` / ``invoke``.
        tools: Workspace toolkit used to execute tool calls.
        messages: Conversation messages seeded with system/human prompts.
        max_tool_calls: Hard cap on tool invocations this iteration.
        max_model_retries: Retries for transient model invoke failures.

    Returns:
        Tuple of ``(messages, tool_call_count, final_text_summary)``.

    Raises:
        CoderLimitError: When the tool-call budget is exhausted.
        Exception: Propagates non-retryable model errors.
    """
    langchain_tools = tools.as_langchain_tools()
    bound = llm.bind_tools(langchain_tools)
    tool_call_count = 0
    final_text = ""

    while True:
        response: Any | None = None
        last_error: Exception | None = None
        for _attempt in range(1, max_model_retries + 1):
            try:
                response = bound.invoke(messages)
                break
            except TimeoutError:
                raise
            except Exception as exc:  # noqa: BLE001 - bounded retry wrapper
                last_error = exc
                logger.warning("coder model invoke failed: %s", exc)
                continue
        if response is None:
            assert last_error is not None
            raise last_error

        messages.append(response)
        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            content = getattr(response, "content", "") or ""
            if isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(str(block.get("text") or ""))
                    else:
                        parts.append(str(block))
                final_text = "\n".join(parts).strip()
            else:
                final_text = str(content).strip()
            break

        for call in tool_calls:
            if tool_call_count >= max_tool_calls:
                raise CoderLimitError(
                    f"exceeded maximum tool calls ({max_tool_calls}) for this iteration"
                )
            tool_call_count += 1
            name = (
                call.get("name")
                if isinstance(call, dict)
                else getattr(call, "name", "")
            )
            call_id = (
                call.get("id") if isinstance(call, dict) else getattr(call, "id", "")
            ) or f"call_{tool_call_count}"
            args = (
                call.get("args")
                if isinstance(call, dict)
                else getattr(call, "args", {})
            ) or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            try:
                result_text = tools.invoke_tool(str(name), dict(args))
            except (
                WorkspaceSecurityError,
                WorkspaceLimitError,
                FileNotFoundError,
                ValueError,
                OSError,
            ) as exc:
                result_text = f"ERROR: {exc}"
            messages.append(ToolMessage(content=result_text, tool_call_id=str(call_id)))

    return messages, tool_call_count, final_text


def compute_manifest_compliance(
    tools: WorkspaceFileTools,
    plan: ProjectPlan,
) -> dict[str, bool]:
    """Map each planned manifest path to whether it exists on disk."""
    existing = set(tools.list_files())
    return {
        spec.path.replace("\\", "/"): spec.path.replace("\\", "/") in existing
        for spec in plan.file_manifest
    }


def build_coder_result(
    *,
    tools: WorkspaceFileTools,
    plan: ProjectPlan,
    findings: list[tuple[str, str]],
    summary: str,
    tool_call_count: int,
) -> CoderResult:
    """Assemble the typed coder result from tools state and findings.

    Resolutions are kept only when this iteration mutated at least one
    file; otherwise proposed resolutions are demoted to unresolved issues.
    """
    compliance = compute_manifest_compliance(tools, plan)
    missing = [path for path, present in compliance.items() if not present]
    unresolved: list[str] = []
    if missing:
        unresolved.append("missing_manifest_files: " + ", ".join(missing))

    mutated = bool(tools.created_files or tools.modified_files or tools.deleted_files)
    resolutions: dict[str, str] = {}
    if mutated:
        resolutions = dict(tools.feedback_resolutions)
    else:
        for finding_id, resolution in tools.feedback_resolutions.items():
            unresolved.append(f"unverified_resolution:{finding_id}:{resolution}")

    finding_ids = {fid for fid, _ in findings}
    for finding_id, description in findings:
        if finding_id not in resolutions:
            unresolved.append(f"unresolved:{finding_id}:{description}")

    # Drop stale resolution keys that were never requested.
    resolutions = {
        key: value
        for key, value in resolutions.items()
        if key in finding_ids or not finding_ids
    }
    if findings:
        resolutions = {
            key: value for key, value in resolutions.items() if key in finding_ids
        }

    if not summary:
        summary = (
            f"Coder completed with {tool_call_count} tool calls; "
            f"created={len(tools.created_files)} "
            f"modified={len(tools.modified_files)} "
            f"deleted={len(tools.deleted_files)}"
        )

    return CoderResult(
        summary=summary,
        created_files=list(tools.created_files),
        modified_files=list(tools.modified_files),
        deleted_files=list(tools.deleted_files),
        unresolved_issues=unresolved,
        feedback_resolutions=resolutions,
        manifest_compliance=compliance,
    )


def _failure_update(
    *,
    iteration: int,
    summary: str,
    error_type: str,
    message: str,
    tools: WorkspaceFileTools | None = None,
    plan: ProjectPlan | None = None,
    unresolved: list[str] | None = None,
) -> dict[str, Any]:
    """Build a coder_failed state update."""
    generated_files = tools.list_files() if tools is not None else []
    file_hashes = tools.file_hashes() if tools is not None else {}
    compliance = (
        compute_manifest_compliance(tools, plan) if tools is not None and plan else {}
    )
    result = CoderResult(
        summary=summary,
        created_files=list(tools.created_files) if tools else [],
        modified_files=list(tools.modified_files) if tools else [],
        deleted_files=list(tools.deleted_files) if tools else [],
        unresolved_issues=list(unresolved or [message]),
        feedback_resolutions=dict(tools.feedback_resolutions) if tools else {},
        manifest_compliance=compliance,
    )
    return {
        "generated_files": generated_files,
        "file_hashes": file_hashes,
        "coder_result": result.model_dump(),
        "iteration": iteration,
        "status": "coder_failed",
        "errors": [{"type": error_type, "message": message}],
    }


def coder_node(
    state: WorkflowState,
    *,
    llm: ToolCallingModel | None = None,
    max_tool_calls: int = MAX_TOOL_CALLS_PER_ITERATION,
    max_file_size: int = DEFAULT_MAX_FILE_SIZE,
    max_project_size: int = DEFAULT_MAX_PROJECT_SIZE,
    max_generated_files: int = DEFAULT_MAX_GENERATED_FILES,
    max_model_retries: int = MAX_MODEL_RETRIES,
) -> dict[str, Any]:
    """Create or revise project files from the validated plan.

    Args:
        state: Workflow state after planning or a change request.
        llm: Optional injectable tool-calling model (for tests).
        max_tool_calls: Per-iteration tool-call budget.
        max_file_size: Maximum bytes per written file.
        max_project_size: Maximum total candidate size in bytes.
        max_generated_files: Maximum file count under candidate/.
        max_model_retries: Bounded retries for model invoke failures.

    Returns:
        State update with ``generated_files``, ``file_hashes``,
        ``coder_result``, incremented ``iteration``, and ``status``.
    """
    current_iteration = int(state.get("iteration") or 0)
    next_iteration = current_iteration + 1
    max_iterations = int(state.get("max_iterations") or MAX_CODER_ITERATIONS)
    if current_iteration >= max_iterations:
        return _failure_update(
            iteration=current_iteration,
            summary="coder iteration limit reached",
            error_type="iteration_limit",
            message=(
                f"iteration {current_iteration} already at max_iterations "
                f"{max_iterations}"
            ),
            unresolved=["iteration_limit"],
        )

    if not state.get("plan"):
        return _failure_update(
            iteration=current_iteration,
            summary="missing plan",
            error_type="invalid_input",
            message="plan is required before coding",
            unresolved=["plan"],
        )

    workspace_path = state.get("workspace_path")
    if not workspace_path:
        raise ValueError("workspace_path is required for coding")

    try:
        plan = ProjectPlan.model_validate(state["plan"])
        validate_plan(plan)
    except (ValidationError, PlanValidationError) as exc:
        return _failure_update(
            iteration=current_iteration,
            summary="invalid plan",
            error_type="invalid_plan",
            message=str(exc),
            unresolved=["invalid_plan"],
        )

    candidate = candidate_dir(workspace_path)
    tools = WorkspaceFileTools(
        candidate,
        max_file_size=max_file_size,
        max_project_size=max_project_size,
        max_generated_files=max_generated_files,
    )
    existing_files = tools.list_files()
    findings = extract_findings(state)
    user_request = str(state.get("user_request") or plan.objective)

    messages: list[Any] = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=build_coder_prompt(
                user_request=user_request,
                plan=plan,
                iteration=current_iteration,
                existing_files=existing_files,
                findings=findings,
            )
        ),
    ]

    model: Any = llm if llm is not None else create_coder_llm()

    try:
        _, tool_call_count, summary = run_tool_loop(
            model,
            tools,
            messages,
            max_tool_calls=max_tool_calls,
            max_model_retries=max_model_retries,
        )
    except CoderLimitError as exc:
        return _failure_update(
            iteration=next_iteration,
            summary="tool-call limit exceeded",
            error_type="tool_limit",
            message=str(exc),
            tools=tools,
            plan=plan,
            unresolved=[str(exc)],
        )
    except TimeoutError as exc:
        return _failure_update(
            iteration=next_iteration,
            summary="coder model timed out",
            error_type="model_timeout",
            message=str(exc),
            tools=tools,
            plan=plan,
        )
    except Exception as exc:  # noqa: BLE001 - map unexpected failures for routing
        logger.exception("coder failed")
        return _failure_update(
            iteration=next_iteration,
            summary="coder failed",
            error_type="coder_error",
            message=str(exc),
            tools=tools,
            plan=plan,
        )

    result = build_coder_result(
        tools=tools,
        plan=plan,
        findings=findings,
        summary=summary,
        tool_call_count=tool_call_count,
    )

    # Missing mandatory manifest files are treated as a soft failure only when
    # nothing was produced; otherwise surface them in unresolved_issues and
    # still advance to verification so deterministic checks can run.
    missing = [path for path, ok in result.manifest_compliance.items() if not ok]
    if missing and not (
        result.created_files or result.modified_files or existing_files
    ):
        return _failure_update(
            iteration=next_iteration,
            summary=result.summary,
            error_type="manifest_incomplete",
            message="missing_manifest_files: " + ", ".join(missing),
            tools=tools,
            plan=plan,
            unresolved=result.unresolved_issues,
        )

    return {
        "generated_files": tools.list_files(),
        "file_hashes": tools.file_hashes(),
        "coder_result": result.model_dump(),
        "iteration": next_iteration,
        "status": "verifying",
        "errors": [],
    }


# Re-export AIMessage for tests that assemble mock tool-call transcripts.
__all__ = [
    "AIMessage",
    "CoderLimitError",
    "MAX_CODER_ITERATIONS",
    "MAX_MODEL_RETRIES",
    "MAX_TOOL_CALLS_PER_ITERATION",
    "build_coder_prompt",
    "build_coder_result",
    "coder_node",
    "create_coder_llm",
    "extract_findings",
    "run_tool_loop",
    "topological_file_order",
]
