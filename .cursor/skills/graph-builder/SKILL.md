---
name: graph-builder
description: >-
  Assemble or modify the LangGraph workflow for the greenfield
  code-generation system. Use this skill for shared workflow state,
  workspace initialization, node registration, checkpointing, human
  interrupts, deterministic routing, iteration limits, and packaging.
  Use it only after the planner, coder, and reviewer node interfaces exist.
  Do not use it to implement agent prompts or agent-specific business logic.
compatibility: Python 3.11+, LangGraph, Pydantic v2
metadata:
  version: "1.1"
  owner: graph-builder
  kind: orchestration-spec
---

# Graph Builder

## Goal

Build the bounded, resumable LangGraph workflow that converts one plain-text software request into a human-approved ZIP archive containing generated project files.

The graph owns orchestration only. Agent-specific behavior remains in the planner, coder, and reviewer modules.

## Use this skill when

Use this skill for changes involving:

- Shared workflow state.
- Graph node registration.
- Conditional edges.
- Workspace initialization.
- Checkpointing and `thread_id` handling.
- LangGraph `interrupt()` approval gates.
- Retry and iteration limits.
- Terminal workflow states.
- Packaging and artifact metadata.
- Graph-level tests.

## Out of scope

Do not implement or duplicate:

- Planner prompts or planning logic.
- Coder prompts, tools, or file-generation logic.
- Reviewer prompts or review logic.
- API endpoints or frontend behavior.
- Model-provider configuration owned by agent modules.

When an imported node contract is missing or incompatible, stop and hand off to the owning skill rather than embedding a replacement in `graph.py`.

## Required dependencies

Before modifying the graph, inspect and reuse:

- `planner_node` input and output contract.
- `coder_node` input and output contract.
- `reviewer_node` input and output contract.
- Verification report schema.
- Human decision schema.
- Existing workspace and packaging utilities.

Do not compile the graph until the imported node interfaces are valid.

## External workflow contract

The only required business input is:

```python
{
    "user_request": str,
}
```

The caller must not be required to supply:

- Repository metadata.
- Existing source files.
- Project structure.
- Language or framework.
- Test commands.
- Workflow ID.
- Workspace path.
- Session ID.

Runtime configuration such as a LangGraph `thread_id` may be supplied through the runnable config, but it is not part of the business payload.

## Required workflow

```text
START
  ↓
initialize_workspace
  ↓
planner
  ↓
coder
  ↓
verify
  ↓
coder_human_gate
  ├── approve ──────────────→ reviewer
  ├── request_changes ──────→ coder
  ├── replan ───────────────→ planner
  └── abort ────────────────→ END

reviewer
  ↓
reviewer_human_gate
  ├── approve ──────────────→ package_project
  ├── request_changes ──────→ coder
  ├── replan ───────────────→ planner
  └── abort ────────────────→ END

package_project
  ↓
END
```

## Agent-count invariant

The workflow contains exactly three LLM-backed agent roles:

1. Planner.
2. Coder.
3. Reviewer.

The following are deterministic workflow nodes and must not be described or implemented as additional agents:

- Workspace initialization.
- Verification.
- Human approval gates.
- Packaging.
- Terminal error handling.

## Shared state contract

Define a typed state with at least:

```python
class WorkflowState(TypedDict, total=False):
    user_request: str

    workflow_id: str
    workspace_path: str

    plan: dict
    planner_feedback: list[str]

    generated_files: list[str]
    file_hashes: dict[str, str]
    coder_result: dict

    verification_report: dict
    review_report: dict

    feedback_history: list[dict]
    coder_human_decision: dict
    reviewer_human_decision: dict

    iteration: int
    max_iterations: int
    status: str

    artifact_path: str | None
    artifact_hash: str | None

    errors: list[dict]
```

### State invariants

- Do not use a message list as the primary state representation.
- Store plans, reports, decisions, and artifact metadata explicitly.
- `workflow_id` is immutable after initialization.
- `workspace_path` is immutable after initialization.
- `iteration` is incremented only when the coder completes an implementation attempt.
- `feedback_history` is append-only.
- Agent nodes return partial state updates only.
- Routing functions are pure and deterministic.

## Workspace initialization

`initialize_workspace` must:

1. Validate that `user_request` is non-empty.
2. Generate a UUID.
3. Create an isolated workflow directory.
4. Create required subdirectories.
5. Initialize state collections and counters.
6. Return a partial state update.

Required layout:

```text
workspaces/
└── <workflow-uuid>/
    ├── candidate/
    ├── snapshots/
    ├── reports/
    └── final/
```

### Workspace invariants

- The coder may write only inside `candidate/`.
- Reviewer access is read-only.
- Snapshots are immutable once created.
- Packaging reads only from the approved candidate snapshot.
- Paths must remain contained within the workflow workspace.

## Checkpointing and resumption

Compile the graph with an injectable checkpointer.

Support:

- `MemorySaver` or equivalent for local development and tests.
- A persistent checkpointer for production.
- A UUID-based `thread_id`.
- Resuming the same workflow after an interrupt.
- Inspecting the persisted state of a paused workflow.

Checkpointing exists for durable state, interruption, recovery, and resumption. Do not describe it as chat-memory storage.

The workflow ID and runnable `thread_id` should be aligned when practical. Their relationship must be documented and tested.

## Human decision schema

Both human gates accept:

```python
class HumanDecision(BaseModel):
    decision: Literal[
        "approve",
        "request_changes",
        "replan",
        "abort",
    ]
    feedback: str = ""
```

Reject unsupported decisions through schema validation.

## Coder human gate

Use LangGraph `interrupt()` after deterministic verification.

Expose:

- Original user request.
- Approved project plan.
- Generated file tree.
- Coder summary.
- Verification report.
- Current iteration.
- Diff from the previous snapshot when available.

Persist the decision in `coder_human_decision` and append a normalized entry to `feedback_history`.

Routing semantics:

- `approve` → reviewer.
- `request_changes` → coder.
- `replan` → planner.
- `abort` → terminal aborted state.

## Reviewer human gate

Use LangGraph `interrupt()` after the reviewer node.

Expose:

- Approved project plan.
- Generated file tree.
- Verification report.
- Reviewer verdict.
- Acceptance-criteria results.
- Findings.
- Residual risks.
- Current iteration.

Persist the decision in `reviewer_human_decision` and append a normalized entry to `feedback_history`.

Routing semantics:

- `approve` → package project.
- `request_changes` → coder.
- `replan` → planner.
- `abort` → terminal aborted state.

The reviewer verdict is advisory. The reviewer human gate owns the final transition.

## Verification node contract

The deterministic verification node must:

1. Create or select an isolated execution environment.
2. Derive commands from approved project metadata, never raw model text.
3. Validate commands against an allowlist.
4. Apply time, process, memory, and output limits.
5. Install dependencies using approved commands.
6. Run syntax or compilation checks.
7. Run formatting checks when configured.
8. Run linting.
9. Run type checking when configured.
10. Run automated tests.
11. Perform startup validation when applicable.
12. Capture stdout, stderr, exit code, duration, and timeout status.
13. Return a structured verification report.

Never execute arbitrary commands emitted by an LLM.

The graph builder owns registration and sequencing of the verification node, not the implementation details of individual verification adapters.

## Routing invariants

Routing must be deterministic and implemented outside agent nodes.

Agent nodes must not:

- Return a successor node.
- Return a LangGraph `Command` that changes routing.
- Add or modify graph edges.
- Bypass mandatory verification or approval nodes.

Required normal transitions:

```text
planner → coder
coder → verify
verify → coder_human_gate
reviewer → reviewer_human_gate
package_project → END
```

## Iteration policy

Define:

```python
MAX_ITERATIONS = 4
```

Initialize:

```python
max_iterations = state.get("max_iterations", MAX_ITERATIONS)
```

Validate that `max_iterations` is within an approved positive range.

Before any transition that returns to `coder` or `planner`, determine whether the implementation-attempt limit has been reached.

When the limit is reached:

- Do not start another coder attempt.
- Set an explicit terminal status such as `max_iterations_reached`.
- Preserve generated files, snapshots, reports, and feedback.
- Return enough metadata for human inspection.
- Never mark the workflow as completed or approved.

No conditional edge may create an unbounded loop.

## Failure states

Represent expected failures explicitly. At minimum support:

```text
invalid_input
planner_failed
coder_failed
verification_failed
reviewer_failed
aborted
max_iterations_reached
packaging_failed
completed
```

A terminal error must include a typed record in `errors` and a non-success status.

Do not silently route to `END` without recording why execution stopped.

## Packaging contract

Run packaging only after final human approval.

The packaging node must:

1. Snapshot or copy the approved candidate workspace.
2. Exclude secrets and temporary artifacts.
3. Create a ZIP archive under `final/`.
4. Calculate a SHA-256 digest.
5. Return artifact metadata.

Exclude at least:

```text
.env
.env.*
*.pem
*.key
__pycache__/
.pytest_cache/
.mypy_cache/
.ruff_cache/
.venv/
venv/
node_modules/
workspaces/*/reports/
```

Successful output must contain:

```python
{
    "status": "completed",
    "workflow_id": "...",
    "artifact_path": "...zip",
    "artifact_hash": "...",
    "generated_files": [...],
    "verification_report": {...},
    "review_report": {...},
}
```

## Preferred module ownership

```text
src/codegen_workflow/
├── graph.py
├── state.py
├── routing.py
├── workspace.py
├── packaging.py
├── nodes/
│   ├── verification.py
│   └── human_gates.py
└── schemas/
    ├── decisions.py
    └── verification.py

tests/
├── test_graph.py
├── test_routing.py
├── test_interrupts.py
├── test_workspace.py
└── test_packaging.py
```

Import planner, coder, and reviewer nodes from their dedicated modules. Do not duplicate those implementations inside graph-owned modules.

## Implementation procedure

1. Inspect all imported node contracts.
2. Define or reconcile the typed shared state.
3. Implement workspace initialization.
4. Implement human decision schemas.
5. Implement deterministic routing functions.
6. Implement interrupt-backed human gates.
7. Register planner, coder, verification, reviewer, and packaging nodes.
8. Add mandatory edges and bounded conditional edges.
9. Compile with an injectable checkpointer.
10. Implement packaging and terminal-state behavior.
11. Add graph-level tests using mocked agents.
12. Run graph, routing, interrupt, workspace, and packaging tests.

## Required tests

Use mocked planner, coder, reviewer, and verification nodes. Do not make live LLM calls.

Add tests for:

1. Plain-text workflow startup.
2. Empty-request rejection.
3. UUID generation.
4. Workspace creation.
5. Initial state values.
6. Planner-to-coder routing.
7. Coder-to-verification routing.
8. Verification-to-coder-gate routing.
9. Coder approval routing to reviewer.
10. Coder change request routing to coder.
11. Coder replan routing to planner.
12. Reviewer approval routing to packaging.
13. Reviewer change request routing to coder.
14. Reviewer replan routing to planner.
15. Abort routing from both gates.
16. Maximum-iteration enforcement from both loop points.
17. Explicit `max_iterations_reached` terminal state.
18. State persistence across an interrupt.
19. Resumption with the same `thread_id`.
20. Rejection of an invalid human decision.
21. Feedback-history append behavior.
22. ZIP packaging.
23. Archive hash calculation.
24. Secret and temporary-file exclusion.
25. Packaging cannot run before final approval.
26. Successful end-to-end execution with mocked agents.
27. Agent nodes cannot bypass mandatory deterministic nodes.
28. Routing functions do not mutate state.

## Definition of done

This skill is complete only when:

- The business input requires only `user_request`.
- The workflow creates its own UUID and workspace.
- Exactly three nodes are LLM-backed agents.
- Agent implementations are imported, not duplicated.
- The coder cannot bypass verification.
- The coder cannot bypass coder human approval.
- The reviewer cannot bypass final human approval.
- Human decisions persist across interrupts.
- The same workflow resumes with the same `thread_id`.
- Routing is deterministic and side-effect free.
- Iterations are bounded.
- Terminal failures are explicit.
- Generated code exists as workspace files.
- Packaging runs only after final approval.
- The final output is a hashed ZIP artifact.
- All graph-level tests pass.
