---
name: reviewer-node
description: >-
  Implement or modify the read-only LangGraph reviewer node for the
  greenfield code-generation workflow. Use this skill when work affects
  reviewer schemas, reviewer prompts, read-only workspace inspection,
  acceptance-criteria evaluation, review findings, reviewer verdicts, or
  reviewer-specific tests. Do not use it for graph routing, file mutation,
  human approval, verification execution, or project packaging.
compatibility: Python 3.11+, LangGraph, Pydantic v2
metadata:
  version: "1.1"
  owner: reviewer-node
  kind: implementation-spec
---

# Reviewer Node

## Goal

Implement an independent, LLM-backed, read-only reviewer node that evaluates a generated project against the approved specification and returns a structured review report.

The reviewer is advisory. It must not mutate generated files and must not select the next graph node.

## Use this skill when

Use this skill for changes involving:

- `ReviewFinding` or `ReviewReport` schemas.
- Reviewer prompts or structured-output configuration.
- Read-only workspace inspection tools.
- Acceptance-criteria evaluation.
- Manifest validation.
- Verification-report interpretation.
- Review verdict rules.
- Reviewer unit tests.

## Out of scope

Do not implement or modify:

- Generated-file creation, editing, or deletion.
- LangGraph conditional edges or routing functions.
- Human approval gates.
- Verification command execution.
- Workspace creation.
- Project packaging.
- API or UI endpoints.

When out-of-scope work is required, stop and hand off to the owning skill.

## Required dependencies

Before implementation, inspect and reuse the existing project definitions for:

- `WorkflowState`.
- `ProjectPlan` and acceptance-criterion identifiers.
- Verification-report schema.
- Workspace path and generated-file conventions.
- Coder result and feedback-resolution format.

Do not duplicate schemas already owned by another module.

## Input contract

The node reads the following fields from `WorkflowState`:

```python
user_request: str
plan: ProjectPlan | dict
generated_files: list[str]
workspace_path: str
verification_report: dict
coder_result: dict
previous_review_report: dict | None
```

### Preconditions

- `user_request` is non-empty.
- `plan` contains stable acceptance-criterion identifiers.
- `workspace_path` resolves to the workflow workspace.
- Every path in `generated_files` is relative to `candidate/`.
- The reviewer may only inspect files under `candidate/`.

If a required input is missing or malformed, return a typed reviewer failure in state. Do not raise an unhandled exception for expected validation errors.

## Output contract

Return only a state update:

```python
{
    "review_report": report.model_dump(),
    "status": "awaiting_reviewer_approval",
}
```

The node must not return a route name, `Command`, successor node, or graph mutation.

## Read-only capability boundary

Expose only the following reviewer tools:

```text
list_files()
read_file(path)
search_files(query)
get_file_hash(path)
read_verification_report()
```

### Tool invariants

- Resolve all paths against `candidate/`.
- Reject absolute paths.
- Reject `..` traversal.
- Reject symlink escapes outside `candidate/`.
- Do not expose shell execution.
- Do not expose file mutation.
- Do not expose graph mutation.

The reviewer must never receive:

```text
write_file
delete_file
move_file
rename_file
shell
subprocess
routing tools
```

## Schemas

Implement `ReviewFinding`:

```python
class ReviewFinding(BaseModel):
    finding_id: str
    severity: Literal["blocking", "major", "minor", "suggestion"]
    category: Literal[
        "requirements",
        "correctness",
        "architecture",
        "security",
        "performance",
        "testing",
        "maintainability",
        "documentation",
    ]
    file: str | None
    line: int | None
    description: str
    evidence: str
    recommendation: str
```

Implement `ReviewReport`:

```python
class ReviewReport(BaseModel):
    verdict: Literal["approve", "request_changes", "replan"]
    acceptance_criteria_results: dict[str, bool]
    manifest_results: dict[str, bool]
    reviewed_files: list[str]
    findings: list[ReviewFinding]
    residual_risks: list[str]
    summary: str
```

### Schema invariants

- `finding_id` is stable within a workflow and deterministic for the same finding when practical.
- Every acceptance criterion from the plan appears exactly once in `acceptance_criteria_results`.
- Every required manifest entry appears exactly once in `manifest_results`.
- `reviewed_files` contains normalized relative paths only.
- Findings reference only inspected files.
- Blocking and major findings include evidence and an actionable recommendation.

## Review procedure

Perform the following sequence:

1. Validate the reviewer input contract.
2. Read the complete approved plan.
3. Read the automated verification report.
4. Enumerate the generated file tree.
5. Validate every required manifest entry.
6. Inspect every file needed to evaluate the plan.
7. Evaluate each acceptance criterion independently.
8. Verify coder claims about resolved prior findings.
9. Identify correctness and requirements defects.
10. Identify security risks.
11. Evaluate test quality, coverage of behavior, and negative cases.
12. Evaluate architecture and maintainability only against the approved plan.
13. Detect missing or incomplete documentation.
14. Produce a structured report.
15. Validate the report before returning it.

Do not infer that a criterion passes solely because the coder claims it does.

## Verdict policy

Return `approve` only when all of the following are true:

- No blocking finding exists.
- No major finding exists.
- All required automated checks pass.
- Every mandatory acceptance criterion passes.
- Every mandatory manifest file exists.
- No unresolved prior blocking or major finding remains.

Return `request_changes` when:

- The approved plan remains valid.
- Implementation changes can resolve the defects without changing the architecture or task decomposition.

Return `replan` when:

- The approved plan is contradictory.
- Required behavior cannot be implemented within the approved architecture.
- Acceptance criteria are incomplete or mutually inconsistent.
- Task decomposition must materially change before coding can continue.

The reviewer verdict is advisory; the human review gate owns the final workflow transition.

## Finding quality rules

A blocking or major finding must contain:

- Concrete evidence.
- A file and line number when available.
- A reproducible scenario, failed check, or failed acceptance criterion.
- A specific corrective action.

Do not report vague findings.

Bad:

```text
Improve code quality.
```

Good:

```text
The task-creation endpoint accepts an empty title because
TaskCreate.title has no minimum-length constraint in app/schemas/task.py.
Add min_length=1 and an API test covering an empty title.
```

## LLM integration requirements

- Use structured output validated by Pydantic.
- Use a bounded model timeout.
- Retry malformed structured output a limited number of times.
- Preserve validation errors in typed reviewer error state.
- Do not make a second model call when deterministic validation can reject the response.
- Keep prompts deterministic and include the plan, verification report, prior findings, and available file-inspection tools.

## Implementation steps

1. Inspect existing state and schema modules.
2. Add or update reviewer schemas in the reviewer-owned schema module.
3. Implement path-safe read-only workspace tools.
4. Implement the reviewer prompt and structured-output model binding.
5. Implement reviewer-node input validation.
6. Execute the review procedure.
7. Validate report completeness and verdict consistency.
8. Return the required state update.
9. Add tests.
10. Run the reviewer test suite and relevant graph integration tests.

## Required tests

Mock the LLM. Do not make live model calls.

Add tests for:

1. Approval of a correct project.
2. Missing required manifest file.
3. Failed mandatory acceptance criterion.
4. Blocking security defect.
5. Major correctness defect.
6. Minor documentation defect without forced rejection.
7. Failed verification check.
8. False coder resolution claim.
9. `request_changes` verdict.
10. `replan` verdict.
11. Every acceptance criterion receives a result.
12. Every manifest requirement receives a result.
13. Stable finding identifiers.
14. Path traversal rejection.
15. Symlink escape rejection.
16. No write or shell tools are exposed.
17. The node returns no routing command.
18. Malformed LLM output is handled without an unhandled exception.

## Definition of done

This skill is complete only when:

- The reviewer has read-only access to `candidate/`.
- Path containment is enforced.
- Every acceptance criterion has an explicit result.
- Every required manifest entry has an explicit result.
- Blocking and major findings contain evidence and actionable recommendations.
- The reviewer distinguishes implementation changes from replanning.
- The reviewer cannot mutate files.
- The reviewer does not control routing.
- The report is Pydantic-validated.
- Reviewer tests pass.
- Relevant graph integration tests pass.
