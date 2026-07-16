"""Workflow service adapting the compiled LangGraph for HTTP handlers."""

from __future__ import annotations

import asyncio
import logging
import threading
from concurrent.futures import TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Any

from langgraph.types import Command

from codegen_workflow.api.config import APISettings
from codegen_workflow.api.errors import (
    ArtifactNotFoundError,
    ArtifactNotReadyError,
    CandidateFileNotFoundError,
    CandidateFileUnreadableError,
    GraphInvocationError,
    InvalidHumanDecisionError,
    InvalidWorkflowTransitionError,
    WorkflowNotFoundError,
    WorkflowTimeoutError,
)
from codegen_workflow.api.schemas import (
    PAUSED_STATUSES,
    TERMINAL_STATUSES,
    CandidateFileContentResponse,
    CandidateFileTreeResponse,
    HumanDecisionRequest,
    RunStatus,
    RunStatusResponse,
    RunTicketResponse,
    RunTraceResponse,
    TraceEvent,
)
from codegen_workflow.graph import run_config_for_thread
from codegen_workflow.routing import MAX_ITERATIONS
from codegen_workflow.tools.workspace import (
    DEFAULT_MAX_FILE_SIZE,
    WorkspaceFileTools,
    WorkspaceSecurityError,
)
from codegen_workflow.workspace import candidate_dir, create_workflow_id

logger = logging.getLogger(__name__)

# Cap preview reads to the same per-file limit used by workspace tools.
_MAX_CANDIDATE_PREVIEW_BYTES = DEFAULT_MAX_FILE_SIZE

_SENSITIVE_DETAIL_KEYS = frozenset(
    {
        "api_key",
        "openai_api_key",
        "authorization",
        "password",
        "secret",
        "token",
        "prompt",
        "system_prompt",
        "raw_prompt",
    }
)


class WorkflowService:
    """Invoke and inspect the compiled workflow graph on behalf of the API.

    Handlers must use this service rather than calling Planner/Coder/Reviewer
    nodes or reproducing graph routing.
    """

    def __init__(
        self,
        graph: Any,
        settings: APISettings,
        *,
        checkpointer: Any | None = None,
    ) -> None:
        """Initialize the service.

        Args:
            graph: Compiled LangGraph workflow.
            settings: Application settings.
            checkpointer: Optional checkpointer reference for readiness checks.
        """
        self.graph = graph
        self.settings = settings
        self.checkpointer = checkpointer
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def _lock_for(self, workflow_id: str) -> threading.Lock:
        """Return a per-workflow lock for resume serialization.

        Args:
            workflow_id: Workflow / thread identifier.

        Returns:
            A :class:`threading.Lock` unique to the workflow.
        """
        with self._locks_guard:
            lock = self._locks.get(workflow_id)
            if lock is None:
                lock = threading.Lock()
                self._locks[workflow_id] = lock
            return lock

    def _invoke_sync(self, payload: Any, config: dict[str, Any]) -> dict[str, Any]:
        """Synchronously invoke the graph with a timeout bound.

        Args:
            payload: Graph input or :class:`Command` resume payload.
            config: Runnable config containing ``thread_id``.

        Returns:
            Graph result dictionary.

        Raises:
            WorkflowTimeoutError: If the invocation exceeds the configured bound.
            GraphInvocationError: If the graph raises an unexpected error.
        """
        workflow_id = (config.get("configurable") or {}).get("thread_id")
        try:
            return self.graph.invoke(
                payload,
                config=config,
                # langgraph invoke doesn't take timeout; bound via wait below
            )
        except TimeoutError as exc:
            raise WorkflowTimeoutError(str(workflow_id) if workflow_id else None) from exc
        except Exception as exc:  # noqa: BLE001 - mapped to safe API error
            logger.exception(
                "graph_invocation_failed",
                extra={
                    "request_id": "-",
                    "workflow_id": str(workflow_id or "-"),
                    "endpoint": "workflow_service",
                    "error_type": type(exc).__name__,
                },
            )
            raise GraphInvocationError(
                workflow_id=str(workflow_id) if workflow_id else None
            ) from exc

    async def _invoke(
        self,
        payload: Any,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """Run a blocking graph invocation off the event loop.

        Args:
            payload: Graph input or resume command.
            config: Runnable config.

        Returns:
            Graph result dictionary.
        """
        workflow_id = (config.get("configurable") or {}).get("thread_id")
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self._invoke_sync, payload, config),
                timeout=self.settings.workflow_timeout_seconds,
            )
        except (TimeoutError, FuturesTimeoutError, asyncio.TimeoutError) as exc:
            raise WorkflowTimeoutError(str(workflow_id) if workflow_id else None) from exc

    def _snapshot(self, workflow_id: str) -> Any:
        """Load the latest checkpoint snapshot for a workflow.

        Args:
            workflow_id: Workflow / thread identifier.

        Returns:
            LangGraph state snapshot.

        Raises:
            WorkflowNotFoundError: If no checkpoint exists for the ID.
        """
        config = run_config_for_thread(workflow_id)
        snapshot = self.graph.get_state(config)
        values = getattr(snapshot, "values", None) or {}
        if not values and not getattr(snapshot, "next", None):
            raise WorkflowNotFoundError(workflow_id)
        if not values.get("workflow_id") and not values.get("user_request"):
            # Empty / unknown thread
            if not getattr(snapshot, "next", None) and not getattr(
                snapshot, "tasks", None
            ):
                raise WorkflowNotFoundError(workflow_id)
        return snapshot

    @staticmethod
    def _interrupt_payload(result: dict[str, Any] | None, snapshot: Any) -> dict[str, Any] | None:
        """Extract the current interrupt payload from a result or snapshot.

        Args:
            result: Optional invoke result dictionary.
            snapshot: Latest state snapshot.

        Returns:
            Interrupt value dictionary, or ``None`` when not paused.
        """
        if result and "__interrupt__" in result:
            interrupts = result["__interrupt__"]
            if interrupts:
                value = getattr(interrupts[0], "value", interrupts[0])
                return value if isinstance(value, dict) else {"value": value}
        interrupts = getattr(snapshot, "interrupts", None) or ()
        if interrupts:
            value = getattr(interrupts[0], "value", interrupts[0])
            return value if isinstance(value, dict) else {"value": value}
        return None

    @staticmethod
    def _pending_gate(snapshot: Any, status: str) -> str | None:
        """Derive the pending human-gate name, if any.

        Args:
            snapshot: Latest state snapshot.
            status: Current workflow status string.

        Returns:
            ``\"coder\"``, ``\"reviewer\"``, or ``None``.
        """
        interrupt = WorkflowService._interrupt_payload(None, snapshot)
        if interrupt and isinstance(interrupt.get("gate"), str):
            return interrupt["gate"]
        if status == "awaiting_coder_approval":
            return "coder"
        if status == "awaiting_reviewer_approval":
            return "reviewer"
        return None

    @staticmethod
    def _public_result(values: dict[str, Any]) -> dict[str, Any]:
        """Build a safe public result summary from workflow state.

        Args:
            values: Workflow state values.

        Returns:
            Redacted summary dictionary.
        """
        return {
            "status": values.get("status"),
            "iteration": values.get("iteration"),
            "max_iterations": values.get("max_iterations"),
            "generated_files": list(values.get("generated_files") or []),
            "plan_summary": {
                "project_name": (values.get("plan") or {}).get("project_name"),
                "language": (values.get("plan") or {}).get("language"),
                "objective": (values.get("plan") or {}).get("objective"),
            }
            if values.get("plan")
            else None,
            "review_verdict": (values.get("review_report") or {}).get("verdict"),
            "verification_passed": (values.get("verification_report") or {}).get(
                "passed"
            ),
            "errors": list(values.get("errors") or []),
            "artifact_hash": values.get("artifact_hash"),
        }

    def _message_for_status(self, status: str, *, interrupted: bool) -> str:
        """Return a client-facing message for a workflow status.

        Args:
            status: Workflow lifecycle status.
            interrupted: Whether the run is paused on a human gate.

        Returns:
            Human-readable status message.
        """
        if interrupted:
            if status == "awaiting_coder_approval":
                return (
                    "Paused at the coder gate. Approve to send the project "
                    "to the automated reviewer."
                )
            if status == "awaiting_reviewer_approval":
                return (
                    "Paused at the reviewer gate. Approve again to package "
                    "the project and unlock the download ZIP."
                )
            return "Workflow paused awaiting human decision."
        messages = {
            "completed": "Workflow completed successfully.",
            "aborted": "Workflow was aborted by human decision.",
            "max_iterations_reached": (
                "Maximum iterations reached; result was not approved."
            ),
            "planner_failed": "Planner failed to produce a valid plan.",
            "coder_failed": "Coder failed to produce a valid implementation.",
            "verification_failed": "Verification failed for the generated project.",
            "reviewer_failed": "Reviewer failed to produce a valid review.",
            "packaging_failed": "Packaging failed for the approved project.",
            "invalid_input": "The ticket was rejected as invalid input.",
            "planning": "Workflow is planning.",
            "coding": "Workflow is generating code.",
            "verifying": "Workflow is verifying generated code.",
            "reviewing": "Workflow is reviewing generated code.",
        }
        return messages.get(status, f"Workflow status: {status}")

    def _build_ticket_response(
        self,
        workflow_id: str,
        result: dict[str, Any] | None,
        snapshot: Any,
    ) -> RunTicketResponse:
        """Map graph state into the public create/resume response.

        Args:
            workflow_id: Workflow identifier.
            result: Optional invoke result.
            snapshot: Latest checkpoint snapshot.

        Returns:
            Populated :class:`RunTicketResponse`.
        """
        values = dict(getattr(snapshot, "values", None) or {})
        if result:
            # Prefer non-interrupt scalar fields from the invoke result.
            for key, value in result.items():
                if key != "__interrupt__":
                    values[key] = value

        interrupt = self._interrupt_payload(result, snapshot)
        status_raw = str(values.get("status") or "planning")
        if interrupt and status_raw not in PAUSED_STATUSES:
            gate = interrupt.get("gate")
            if gate == "coder":
                status_raw = "awaiting_coder_approval"
            elif gate == "reviewer":
                status_raw = "awaiting_reviewer_approval"

        validated_status = self._coerce_status(status_raw)

        artifact_url = None
        if validated_status == "completed" and values.get("artifact_path"):
            artifact_url = f"/runs/{workflow_id}/artifact"

        interrupted = interrupt is not None
        message = self._message_for_status(validated_status, interrupted=interrupted)
        public_result = (
            self._public_result(values)
            if validated_status in TERMINAL_STATUSES or interrupt is None
            else None
        )

        return RunTicketResponse(
            workflow_id=workflow_id,
            status=validated_status,
            message=message,
            interrupt=interrupt,
            result=public_result,
            artifact_url=artifact_url,
            trace_url=f"/runs/{workflow_id}/trace",
        )

    @staticmethod
    def _coerce_status(status_raw: str) -> RunStatus:
        """Coerce an arbitrary status string into a supported RunStatus.

        Args:
            status_raw: Raw status from workflow state.

        Returns:
            A valid :data:`RunStatus` literal value.
        """
        allowed: set[str] = {
            "planning",
            "coding",
            "verifying",
            "awaiting_coder_approval",
            "reviewing",
            "awaiting_reviewer_approval",
            "completed",
            "aborted",
            "max_iterations_reached",
            "planner_failed",
            "coder_failed",
            "verification_failed",
            "reviewer_failed",
            "packaging_failed",
            "invalid_input",
        }
        if status_raw in allowed:
            return status_raw  # type: ignore[return-value]
        return "coding"

    async def start_ticket(
        self,
        ticket: str,
        max_iterations: int | None = None,
    ) -> tuple[RunTicketResponse, int]:
        """Start a new workflow from a free-text ticket.

        Args:
            ticket: Stripped user software request.
            max_iterations: Optional bounded iteration limit.

        Returns:
            Tuple of response model and suggested HTTP status code.
        """
        workflow_id = create_workflow_id()
        config = run_config_for_thread(workflow_id)
        graph_input: dict[str, Any] = {"user_request": ticket}
        if max_iterations is not None:
            graph_input["max_iterations"] = max_iterations

        logger.info(
            "workflow_created ticket_length=%s",
            len(ticket),
            extra={
                "request_id": "-",
                "workflow_id": workflow_id,
                "endpoint": "POST /run-ticket",
            },
        )

        result = await self._invoke(graph_input, config)
        snapshot = self.graph.get_state(config)
        response = self._build_ticket_response(workflow_id, result, snapshot)

        if response.interrupt is not None or response.status in PAUSED_STATUSES:
            return response, 202
        return response, 201

    async def submit_decision(
        self,
        workflow_id: str,
        decision: HumanDecisionRequest,
    ) -> tuple[RunTicketResponse, int]:
        """Resume a paused workflow with a human decision.

        Args:
            workflow_id: Existing workflow / thread identifier.
            decision: Validated human decision payload.

        Returns:
            Tuple of response model and HTTP status code.

        Raises:
            WorkflowNotFoundError: Unknown workflow.
            InvalidWorkflowTransitionError: Not paused or terminal.
            InvalidHumanDecisionError: Decision invalid for the pending gate.
        """
        lock = self._lock_for(workflow_id)
        if not lock.acquire(blocking=False):
            raise InvalidWorkflowTransitionError(
                workflow_id,
                "A decision is already being processed for this workflow.",
                code="duplicate_decision",
            )
        try:
            snapshot = self._snapshot(workflow_id)
            values = dict(snapshot.values or {})
            status = str(values.get("status") or "")
            interrupt = self._interrupt_payload(None, snapshot)

            if status in TERMINAL_STATUSES and interrupt is None:
                raise InvalidWorkflowTransitionError(
                    workflow_id,
                    "A completed or terminal workflow cannot be resumed.",
                )
            if interrupt is None:
                raise InvalidWorkflowTransitionError(
                    workflow_id,
                    "No human interrupt is pending for this workflow.",
                    code="no_interrupt",
                )

            gate = interrupt.get("gate")
            # Decision validity for gate is structural; all listed decisions are
            # accepted by both gates. Wrong-gate protection is satisfied by the
            # interrupt being attached to a specific gate node.
            if gate not in {"coder", "reviewer"}:
                raise InvalidHumanDecisionError(
                    "Decision submitted for an unrecognized gate.",
                    workflow_id=workflow_id,
                )

            config = run_config_for_thread(workflow_id)
            resume_payload = {
                "decision": decision.decision,
                "feedback": decision.feedback.strip(),
            }
            logger.info(
                "human_resume gate=%s decision=%s",
                gate,
                decision.decision,
                extra={
                    "request_id": "-",
                    "workflow_id": workflow_id,
                    "endpoint": "POST /runs/{id}/decision",
                },
            )
            result = await self._invoke(Command(resume=resume_payload), config)
            snapshot = self.graph.get_state(config)
            response = self._build_ticket_response(workflow_id, result, snapshot)
            if response.interrupt is not None or response.status in PAUSED_STATUSES:
                return response, 202
            return response, 200
        finally:
            lock.release()

    def get_status(self, workflow_id: str) -> RunStatusResponse:
        """Return the current persisted workflow status.

        Args:
            workflow_id: Workflow identifier.

        Returns:
            :class:`RunStatusResponse` for the run.
        """
        snapshot = self._snapshot(workflow_id)
        values = dict(snapshot.values or {})
        status_raw = self._coerce_status(str(values.get("status") or "planning"))
        pending = self._pending_gate(snapshot, status_raw)
        artifact_url = None
        if status_raw == "completed" and values.get("artifact_path"):
            artifact_url = f"/runs/{workflow_id}/artifact"
        created_at = getattr(snapshot, "created_at", None)
        return RunStatusResponse(
            workflow_id=str(values.get("workflow_id") or workflow_id),
            status=status_raw,  # type: ignore[arg-type]
            iteration=int(values.get("iteration") or 0),
            max_iterations=int(values.get("max_iterations") or MAX_ITERATIONS),
            pending_gate=pending,
            generated_files=list(values.get("generated_files") or []),
            artifact_url=artifact_url,
            created_at=str(created_at) if created_at else None,
            updated_at=str(created_at) if created_at else None,
        )

    @staticmethod
    def _redact_details(details: dict[str, Any] | None) -> dict[str, Any] | None:
        """Drop sensitive keys from trace details.

        Args:
            details: Raw detail dictionary.

        Returns:
            Redacted dictionary or ``None``.
        """
        if not details:
            return None
        cleaned: dict[str, Any] = {}
        for key, value in details.items():
            if key.lower() in _SENSITIVE_DETAIL_KEYS:
                continue
            if isinstance(value, str) and any(
                token in key.lower() for token in ("prompt", "secret", "key", "token")
            ):
                continue
            cleaned[key] = value
        return cleaned or None

    def get_trace(self, workflow_id: str) -> RunTraceResponse:
        """Build a redacted trace from structured workflow state.

        Args:
            workflow_id: Workflow identifier.

        Returns:
            Ordered :class:`RunTraceResponse`.
        """
        snapshot = self._snapshot(workflow_id)
        values = dict(snapshot.values or {})
        events: list[TraceEvent] = []
        seq = 0

        def add(
            node: str,
            status: str,
            summary: str,
            *,
            iteration: int | None = None,
            details: dict[str, Any] | None = None,
        ) -> None:
            nonlocal seq
            seq += 1
            events.append(
                TraceEvent(
                    sequence=seq,
                    node=node,
                    status=status,
                    iteration=iteration,
                    summary=summary,
                    timestamp=None,
                    details=self._redact_details(details),
                )
            )

        if values.get("workspace_path") or values.get("workflow_id"):
            add(
                "initialize_workspace",
                "ok",
                "Workspace initialized for workflow.",
                details={"workflow_id": workflow_id},
            )

        planner_errors = values.get("planner_errors") or []
        if values.get("plan"):
            add(
                "planner",
                "completed",
                "Planner produced a project plan.",
                details={
                    "project_name": (values.get("plan") or {}).get("project_name"),
                    "language": (values.get("plan") or {}).get("language"),
                },
            )
        elif planner_errors or values.get("status") == "planner_failed":
            add(
                "planner",
                "failed",
                "Planner failed.",
                details={"error_count": len(planner_errors)},
            )

        iteration = int(values.get("iteration") or 0)
        if iteration > 0 or values.get("coder_result"):
            add(
                "coder",
                "completed" if values.get("coder_result") else str(values.get("status")),
                (values.get("coder_result") or {}).get("summary")
                or "Coder iteration recorded.",
                iteration=iteration or None,
                details={
                    "generated_files": list(values.get("generated_files") or []),
                },
            )

        verification = values.get("verification_report") or {}
        if verification:
            add(
                "verify",
                "passed" if verification.get("passed") else "failed",
                "Verification completed.",
                iteration=iteration or None,
                details={
                    "overall_status": verification.get("overall_status"),
                    "passed": verification.get("passed"),
                },
            )

        for entry in values.get("feedback_history") or []:
            if not isinstance(entry, dict):
                continue
            gate = str(entry.get("gate") or "human")
            add(
                f"{gate}_human_gate",
                str(entry.get("decision") or "decision"),
                f"Human decision at {gate} gate: {entry.get('decision')}.",
                iteration=entry.get("iteration"),
                details={
                    "decision": entry.get("decision"),
                    "feedback_length": len(str(entry.get("feedback") or "")),
                },
            )

        interrupt = self._interrupt_payload(None, snapshot)
        if interrupt:
            gate = interrupt.get("gate", "human")
            add(
                f"{gate}_human_gate",
                "interrupted",
                f"Workflow interrupted at {gate} gate.",
                iteration=interrupt.get("iteration"),
                details={"gate": gate},
            )

        review = values.get("review_report") or {}
        if review:
            add(
                "reviewer",
                str(review.get("verdict") or "reviewed"),
                review.get("summary") or "Reviewer produced a report.",
                iteration=iteration or None,
                details={
                    "verdict": review.get("verdict"),
                    "finding_count": len(review.get("findings") or []),
                },
            )

        status = str(values.get("status") or "")
        if status == "max_iterations_reached":
            add(
                "routing",
                "max_iterations_reached",
                "Retry limit exhausted; result was not approved.",
                iteration=iteration or None,
            )
        if status == "completed" or values.get("artifact_path"):
            add(
                "package_project",
                "completed" if status == "completed" else status,
                "Packaging finished."
                if status == "completed"
                else "Packaging outcome recorded.",
                details={"artifact_hash": values.get("artifact_hash")},
            )
        elif status == "packaging_failed":
            add(
                "package_project",
                "failed",
                "Packaging failed.",
            )
        if status in TERMINAL_STATUSES and status not in {
            "completed",
            "max_iterations_reached",
            "packaging_failed",
        }:
            add(
                "workflow",
                status,
                f"Workflow reached terminal status: {status}.",
                iteration=iteration or None,
            )

        return RunTraceResponse(workflow_id=workflow_id, events=events)

    def _candidate_root(self, workflow_id: str) -> Path:
        """Resolve the candidate directory for a workflow with containment checks.

        Args:
            workflow_id: Workflow identifier.

        Returns:
            Absolute path to ``candidate/`` under the workflow workspace.

        Raises:
            WorkflowNotFoundError: Unknown workflow.
            InvalidWorkflowTransitionError: Missing workspace or path escape.
        """
        snapshot = self._snapshot(workflow_id)
        values = dict(snapshot.values or {})
        workspace_path = values.get("workspace_path")
        if not workspace_path:
            raise InvalidWorkflowTransitionError(
                workflow_id,
                "Workspace has not been initialized for this workflow.",
                code="workspace_not_ready",
            )
        workspace_root = Path(workspace_path).resolve()
        base = self.settings.workspace_base_dir.resolve()
        if not self._is_relative_to(workspace_root, base):
            raise InvalidWorkflowTransitionError(
                workflow_id,
                "Workspace path is outside the configured workspace directory.",
                code="workspace_path_violation",
            )
        candidate = candidate_dir(workspace_root)
        if not self._is_relative_to(candidate.resolve(), workspace_root):
            raise InvalidWorkflowTransitionError(
                workflow_id,
                "Candidate path escaped the workflow workspace.",
                code="workspace_path_violation",
            )
        return candidate.resolve()

    def list_candidate_files(self, workflow_id: str) -> CandidateFileTreeResponse:
        """List relative file paths under the workflow candidate directory.

        Args:
            workflow_id: Workflow identifier.

        Returns:
            Sorted relative POSIX paths for generated candidate files.
        """
        root = self._candidate_root(workflow_id)
        if not root.is_dir():
            return CandidateFileTreeResponse(workflow_id=workflow_id, files=[])
        files: list[str] = []
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            try:
                resolved = path.resolve()
                if not self._is_relative_to(resolved, root):
                    continue
                files.append(resolved.relative_to(root).as_posix())
            except (OSError, ValueError):
                continue
        return CandidateFileTreeResponse(workflow_id=workflow_id, files=files)

    def read_candidate_file(
        self,
        workflow_id: str,
        relative_path: str,
    ) -> CandidateFileContentResponse:
        """Read one UTF-8 candidate file for human review.

        Args:
            workflow_id: Workflow identifier.
            relative_path: Path relative to ``candidate/``.

        Returns:
            File path, UTF-8 content, and size metadata.

        Raises:
            CandidateFileNotFoundError: Missing file.
            CandidateFileUnreadableError: Binary, oversized, or empty path.
            InvalidWorkflowTransitionError: Path escape / traversal.
        """
        root = self._candidate_root(workflow_id)
        if not root.is_dir():
            raise CandidateFileNotFoundError(workflow_id, relative_path)

        # Construct tools against an existing root without relying on mkdir
        # for path resolution; WorkspaceFileTools creates the root when
        # missing, so only instantiate after the directory check above.
        tools = WorkspaceFileTools(root, max_file_size=_MAX_CANDIDATE_PREVIEW_BYTES)
        try:
            target = tools.resolve_path(relative_path)
        except WorkspaceSecurityError as exc:
            raise InvalidWorkflowTransitionError(
                workflow_id,
                "Requested path is not allowed inside the candidate workspace.",
                code="file_path_violation",
                details=[str(exc)],
            ) from exc

        if not target.is_file():
            raise CandidateFileNotFoundError(workflow_id, relative_path)

        size = target.stat().st_size
        if size > _MAX_CANDIDATE_PREVIEW_BYTES:
            raise CandidateFileUnreadableError(
                workflow_id,
                "File exceeds the maximum preview size.",
                details=[
                    f"size_bytes={size}",
                    f"max_bytes={_MAX_CANDIDATE_PREVIEW_BYTES}",
                ],
            )

        try:
            content = target.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise CandidateFileUnreadableError(
                workflow_id,
                "File is not valid UTF-8 text and cannot be previewed.",
                details=[relative_path],
            ) from exc

        return CandidateFileContentResponse(
            workflow_id=workflow_id,
            path=target.resolve().relative_to(root).as_posix(),
            content=content,
            encoding="utf-8",
            size_bytes=size,
        )

    def resolve_artifact_path(self, workflow_id: str) -> Path:
        """Resolve and validate the artifact ZIP path for a workflow.

        Args:
            workflow_id: Workflow identifier.

        Returns:
            Absolute path to the ZIP archive.

        Raises:
            WorkflowNotFoundError: Unknown workflow.
            ArtifactNotReadyError: Workflow not completed.
            ArtifactNotFoundError: Missing artifact file.
            InvalidWorkflowTransitionError: Path escapes allowed roots.
        """
        snapshot = self._snapshot(workflow_id)
        values = dict(snapshot.values or {})
        status = str(values.get("status") or "")
        if status != "completed":
            raise ArtifactNotReadyError(workflow_id)

        raw_path = values.get("artifact_path")
        if not raw_path:
            raise ArtifactNotFoundError(workflow_id)

        artifact_path = Path(raw_path).resolve()
        if not artifact_path.is_file():
            raise ArtifactNotFoundError(workflow_id)

        allowed_roots = [
            self.settings.workspace_base_dir.resolve(),
            self.settings.artifact_base_dir.resolve(),
        ]
        workspace_path = values.get("workspace_path")
        if workspace_path:
            allowed_roots.append(Path(workspace_path).resolve())

        if not any(
            self._is_relative_to(artifact_path, root) for root in allowed_roots
        ):
            raise InvalidWorkflowTransitionError(
                workflow_id,
                "Artifact path is outside the configured directories.",
                code="artifact_path_violation",
            )

        # Confirm the artifact is associated with this workflow ID.
        if workflow_id not in artifact_path.name and workflow_id not in str(
            artifact_path
        ):
            raise InvalidWorkflowTransitionError(
                workflow_id,
                "Artifact does not belong to the requested workflow.",
                code="artifact_mismatch",
            )
        return artifact_path

    @staticmethod
    def _is_relative_to(path: Path, root: Path) -> bool:
        """Return whether ``path`` is under ``root``.

        Args:
            path: Candidate absolute path.
            root: Allowed root directory.

        Returns:
            ``True`` when ``path`` is inside ``root``.
        """
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    def readiness_checks(self) -> dict[str, bool]:
        """Evaluate readiness dependencies without calling an LLM.

        Returns:
            Mapping of check name to boolean pass/fail.
        """
        workspace_ok = False
        try:
            self.settings.workspace_base_dir.mkdir(parents=True, exist_ok=True)
            probe = self.settings.workspace_base_dir / ".write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            workspace_ok = True
        except OSError:
            workspace_ok = False

        artifact_ok = False
        try:
            self.settings.artifact_base_dir.mkdir(parents=True, exist_ok=True)
            artifact_ok = self.settings.artifact_base_dir.is_dir()
        except OSError:
            artifact_ok = False

        env_ok = True
        if self.settings.is_production:
            env_ok = bool(self.settings.openai_api_key)

        return {
            "application_initialized": True,
            "graph_compiled": self.graph is not None,
            "checkpointer_available": self.checkpointer is not None
            or self.graph is not None,
            "workspace_writable": workspace_ok,
            "artifact_directory": artifact_ok,
            "environment_configured": env_ok,
        }
