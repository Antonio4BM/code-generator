"""Deterministic verification node for generated candidate projects.

This node is not an LLM agent. It creates an isolated execution context,
runs allowlisted install and validation commands with resource limits,
and returns a structured verification report.
"""

from __future__ import annotations

import os
import resource
import subprocess
import time
from pathlib import Path
from typing import Any, Sequence

from codegen_workflow.schemas.verification import CommandResult, VerificationReport
from codegen_workflow.state import WorkflowState
from codegen_workflow.workspace import candidate_dir, reports_dir

# Only these executable names may appear as command[0] after resolution.
ALLOWED_EXECUTABLES = frozenset(
    {
        "python",
        "python3",
        "pip",
        "pip3",
        "pytest",
        "ruff",
        "mypy",
        "black",
        "flake8",
        "uv",
        "npm",
        "node",
        "pnpm",
        "yarn",
        "poetry",
        "hatch",
        "pdm",
        "tox",
        "make",
        "cargo",
        "go",
        "javac",
        "java",
        "mvn",
        "gradle",
    }
)

# Default resource limits for subprocesses (soft, hard) in bytes / seconds.
DEFAULT_CPU_SECONDS = 60
DEFAULT_AS_BYTES = 512 * 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 120


def _resolve_executable(name: str) -> str | None:
    """Resolve an executable on PATH if it is allowlisted.

    Args:
        name: Executable basename or path basename.

    Returns:
        Absolute path string when found and allowed, else ``None``.
    """
    basename = Path(name).name
    if basename not in ALLOWED_EXECUTABLES:
        return None
    search_paths = os.environ.get("PATH", "").split(os.pathsep)
    for directory in search_paths:
        candidate = Path(directory) / basename
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    # Fall back to the basename for relative resolution by the shell-less
    # subprocess when PATH lookup failed but the name is allowlisted.
    return basename if basename in ALLOWED_EXECUTABLES else None


def validate_command(command: Sequence[str]) -> list[str]:
    """Validate and normalize a command against the allowlist.

    Args:
        command: Proposed argv list from the plan or configuration.

    Returns:
        Normalized argv with a resolved executable path.

    Raises:
        ValueError: If the command is empty or not allowlisted.
    """
    if not command:
        raise ValueError("Command must be a non-empty list")
    resolved = _resolve_executable(str(command[0]))
    if resolved is None:
        raise ValueError(f"Command not allowlisted: {command[0]!r}")
    return [resolved, *[str(part) for part in command[1:]]]


def _preexec_limits() -> None:
    """Apply CPU and address-space limits in the child process."""
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (DEFAULT_CPU_SECONDS, DEFAULT_CPU_SECONDS))
    except (ValueError, OSError):
        pass
    try:
        resource.setrlimit(resource.RLIMIT_AS, (DEFAULT_AS_BYTES, DEFAULT_AS_BYTES))
    except (ValueError, OSError):
        pass


def run_command(
    name: str,
    command: Sequence[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> CommandResult:
    """Run one allowlisted command and capture structured output.

    Args:
        name: Logical step name recorded in the report.
        command: Proposed argv; validated before execution.
        cwd: Working directory for the subprocess.
        env: Optional environment overrides merged onto the current env.
        timeout: Maximum wall-clock seconds before the process is killed.

    Returns:
        :class:`CommandResult` describing the execution outcome.
    """
    started = time.perf_counter()
    try:
        argv = validate_command(command)
    except ValueError as exc:
        return CommandResult(
            name=name,
            command=list(command),
            exit_code=127,
            stderr=str(exc),
            duration_seconds=0.0,
        )

    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    # Keep verification hermetic: do not inherit API keys into tests blindly.
    for key in list(merged_env):
        if key.endswith("_API_KEY") or key in {"OPENAI_API_KEY", "ANTHROPIC_API_KEY"}:
            merged_env.pop(key, None)

    try:
        completed = subprocess.run(
            argv,
            cwd=str(cwd),
            env=merged_env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            preexec_fn=_preexec_limits,
        )
        return CommandResult(
            name=name,
            command=argv,
            exit_code=completed.returncode,
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
            duration_seconds=time.perf_counter() - started,
        )
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            name=name,
            command=argv,
            exit_code=124,
            stdout=(exc.stdout or "") if isinstance(exc.stdout, str) else "",
            stderr=f"Timed out after {timeout}s",
            duration_seconds=time.perf_counter() - started,
        )
    except OSError as exc:
        return CommandResult(
            name=name,
            command=argv,
            exit_code=126,
            stderr=str(exc),
            duration_seconds=time.perf_counter() - started,
        )


def _default_commands(plan: dict[str, Any]) -> list[tuple[str, list[str]]]:
    """Build the ordered verification command list from the plan.

    Args:
        plan: Project plan dictionary from planner output.

    Returns:
        Ordered ``(name, argv)`` pairs for verification steps.
    """
    commands: list[tuple[str, list[str]]] = []

    for index, install in enumerate(plan.get("install_commands") or []):
        if isinstance(install, str):
            argv = install.split()
        else:
            argv = list(install)
        commands.append((f"install_{index}", argv))

    # Syntax / compilation check for Python projects when applicable.
    language = str(plan.get("language") or "").lower()
    if language in {"", "python", "py"}:
        commands.append(("syntax", ["python3", "-m", "compileall", "-q", "."]))

    for index, validation in enumerate(plan.get("validation_commands") or []):
        if isinstance(validation, str):
            argv = validation.split()
        else:
            argv = list(validation)
        commands.append((f"validate_{index}", argv))

    if not any(name.startswith("validate_") for name, _ in commands):
        commands.append(("tests", ["python3", "-m", "pytest", "-q"]))

    return commands


def build_verification_report(
    workspace_path: Path | str,
    plan: dict[str, Any] | None = None,
    *,
    command_runner=run_command,
) -> VerificationReport:
    """Execute verification steps against the candidate project.

    Args:
        workspace_path: Workflow workspace root.
        plan: Optional project plan providing install/validation commands.
        command_runner: Callable used to execute commands (injectable for
            tests). Signature matches :func:`run_command`.

    Returns:
        Structured :class:`VerificationReport`.
    """
    plan = plan or {}
    candidate = candidate_dir(workspace_path)
    candidate.mkdir(parents=True, exist_ok=True)

    results: list[CommandResult] = []
    errors: list[dict[str, Any]] = []

    for name, argv in _default_commands(plan):
        result = command_runner(name, argv, cwd=candidate)
        results.append(result)
        if result.exit_code != 0:
            errors.append(
                {
                    "step": name,
                    "exit_code": result.exit_code,
                    "stderr": result.stderr[-2000:],
                }
            )

    passed = all(item.exit_code == 0 for item in results) if results else False
    report = VerificationReport(
        passed=passed,
        overall_status="passed" if passed else "failed",
        commands=results,
        errors=errors,
        metadata={"candidate_path": str(candidate)},
    )

    report_dir = reports_dir(workspace_path)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "verification.json").write_text(
        report.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return report


def verification_node(state: WorkflowState) -> dict[str, Any]:
    """LangGraph node that runs deterministic project verification.

    Model-generated commands are never executed without allowlist
    validation. Resource limits and timeouts bound each subprocess.

    Args:
        state: Workflow state after the coder node.

    Returns:
        State update with ``verification_report`` and status.

    Raises:
        ValueError: If ``workspace_path`` is missing.
    """
    workspace_path = state.get("workspace_path")
    if not workspace_path:
        raise ValueError("workspace_path is required for verification")

    report = build_verification_report(
        workspace_path,
        plan=state.get("plan") or {},
    )
    return {
        "verification_report": report.model_dump(),
        "status": "awaiting_coder_approval",
    }
