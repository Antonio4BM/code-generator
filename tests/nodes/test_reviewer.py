"""Unit tests for the read-only reviewer node (mocked LLM)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from codegen_workflow.nodes.reviewer import (
    collect_acceptance_criteria,
    reviewer_node,
)
from codegen_workflow.schemas.plan import ProjectPlan
from codegen_workflow.schemas.review import ReviewFinding, ReviewReport
from codegen_workflow.tools.readonly import (
    FORBIDDEN_MUTATION_TOOL_NAMES,
    READ_ONLY_TOOL_NAMES,
    ReadOnlyWorkspaceTools,
)
from codegen_workflow.workspace import create_workspace


def _valid_plan_dict(**overrides: Any) -> dict[str, Any]:
    """Minimal valid plan for reviewer tests."""
    plan: dict[str, Any] = {
        "project_name": "demo_app",
        "objective": "Deliver a small demonstrable application.",
        "assumptions": ["Network access is available for package installs."],
        "language": "python",
        "framework": None,
        "architecture_pattern": "modular monolith",
        "dependencies": ["pytest"],
        "epics": [
            {
                "id": "E1",
                "title": "Core delivery",
                "description": "Ship the core application behavior.",
                "acceptance_criteria": [
                    "All stories under this epic are implemented and tested."
                ],
            }
        ],
        "stories": [
            {
                "id": "S1",
                "epic_id": "E1",
                "title": "Primary capability",
                "description": (
                    "As a user, I want the core feature, so that I can complete "
                    "the primary task."
                ),
                "acceptance_criteria": [
                    "Running the app with valid input exits with code 0."
                ],
            }
        ],
        "tasks": [
            {
                "id": "T1",
                "story_id": "S1",
                "title": "Add dependency config",
                "description": "Create dependency configuration for the project.",
                "task_type": "configuration",
                "dependencies": [],
                "files": ["pyproject.toml"],
                "acceptance_criteria": [
                    "pyproject.toml lists pytest as a development dependency."
                ],
            },
            {
                "id": "T2",
                "story_id": "S1",
                "title": "Implement source module",
                "description": "Implement the primary application module.",
                "task_type": "source",
                "dependencies": ["T1"],
                "files": ["src/demo_app/main.py"],
                "acceptance_criteria": [
                    "main.py exposes a callable entrypoint named main."
                ],
            },
            {
                "id": "T3",
                "story_id": "S1",
                "title": "Add automated tests",
                "description": "Cover the primary source behavior with tests.",
                "task_type": "test",
                "dependencies": ["T2"],
                "files": ["tests/test_main.py"],
                "acceptance_criteria": [
                    "pytest reports at least one passing test for main."
                ],
            },
            {
                "id": "T4",
                "story_id": "S1",
                "title": "Add documentation",
                "description": "Document install and run steps.",
                "task_type": "documentation",
                "dependencies": ["T1"],
                "files": ["README.md"],
                "acceptance_criteria": [
                    "README.md includes install and run command sections."
                ],
            },
        ],
        "file_manifest": [
            {
                "path": "pyproject.toml",
                "purpose": "Declare project metadata and dependencies.",
                "file_type": "configuration",
                "requirements": ["Include pytest dependency."],
                "depends_on": [],
            },
            {
                "path": "src/demo_app/main.py",
                "purpose": "Application entrypoint.",
                "file_type": "source",
                "requirements": ["Expose main()."],
                "depends_on": ["pyproject.toml"],
            },
            {
                "path": "tests/test_main.py",
                "purpose": "Automated tests for main.",
                "file_type": "test",
                "requirements": ["Assert main() runs successfully."],
                "depends_on": ["src/demo_app/main.py"],
            },
            {
                "path": "README.md",
                "purpose": "Project documentation.",
                "file_type": "documentation",
                "requirements": ["Document install and run."],
                "depends_on": [],
            },
        ],
        "install_commands": ["pip install -e '.[dev]'"],
        "validation_commands": ["pytest -q"],
        "run_command": "python -m demo_app",
        "risks": ["Scope may expand if requirements change."],
    }
    plan.update(overrides)
    return plan


def _seed_complete_project(candidate: Path) -> None:
    """Write a complete candidate tree matching the sample plan."""
    files = {
        "pyproject.toml": "[project]\nname='demo_app'\ndependencies=[]\n\n[project.optional-dependencies]\ndev=['pytest']\n",
        "src/demo_app/main.py": "def main():\n    return 0\n",
        "tests/test_main.py": (
            "from demo_app.main import main\n\ndef test_main():\n    assert main() == 0\n"
        ),
        "README.md": "# demo_app\n\n## Install\npip install -e .[dev]\n\n## Run\npython -m demo_app\n",
    }
    for relative, content in files.items():
        path = candidate / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def _all_criteria_true(plan: dict[str, Any]) -> dict[str, bool]:
    """Mark every collected acceptance criterion as passing."""
    keys = collect_acceptance_criteria(ProjectPlan.model_validate(plan))
    return {key: True for key in keys}


def _manifest_true(plan: dict[str, Any]) -> dict[str, bool]:
    """Mark every manifest path as present."""
    return {item["path"]: True for item in plan["file_manifest"]}


class ScriptedReviewerLLM:
    """Mock reviewer model: optional tool calls, then structured report."""

    def __init__(
        self,
        report: ReviewReport,
        *,
        tool_script: list[AIMessage] | None = None,
    ) -> None:
        self.report = report
        self.tool_script = list(tool_script or [])
        self.tool_calls = 0
        self.structured_invokes = 0

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedReviewerLLM:
        return self

    def invoke(
        self, messages: Any, config: Any | None = None, **kwargs: Any
    ) -> AIMessage:
        if self.tool_calls < len(self.tool_script):
            message = self.tool_script[self.tool_calls]
            self.tool_calls += 1
            return message
        return AIMessage(content="inspection complete")

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        parent = self

        class _Structured:
            def invoke(
                self, messages: Any, config: Any | None = None, **kw: Any
            ) -> ReviewReport:
                parent.structured_invokes += 1
                return parent.report

        return _Structured()


@pytest.fixture
def workspace(tmp_path: Path) -> tuple[str, Path]:
    """Create an isolated workflow workspace."""
    return create_workspace(base_dir=tmp_path / "workspaces")


def _base_state(root: Path, plan: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    """Build a reviewer state payload with sensible defaults."""
    state: dict[str, Any] = {
        "user_request": "Build demo_app",
        "plan": plan,
        "workspace_path": str(root),
        "generated_files": [item["path"] for item in plan["file_manifest"]],
        "verification_report": {
            "passed": True,
            "overall_status": "passed",
            "commands": [{"name": "pytest", "exit_code": 0, "stderr": ""}],
        },
        "coder_result": {
            "summary": "generated",
            "created_files": [item["path"] for item in plan["file_manifest"]],
            "modified_files": [],
            "deleted_files": [],
            "unresolved_issues": [],
            "feedback_resolutions": {},
            "manifest_compliance": _manifest_true(plan),
        },
        "review_report": {},
        "previous_review_report": {},
    }
    state.update(overrides)
    return state


def test_correct_project_approval(workspace: tuple[str, Path]) -> None:
    """A complete, verified project can receive an approve verdict."""
    _, root = workspace
    _seed_complete_project(root / "candidate")
    plan = _valid_plan_dict()
    report = ReviewReport(
        verdict="approve",
        acceptance_criteria_results=_all_criteria_true(plan),
        manifest_results=_manifest_true(plan),
        reviewed_files=list(_manifest_true(plan)),
        findings=[],
        residual_risks=[],
        summary="All criteria and checks passed.",
    )
    result = reviewer_node(_base_state(root, plan), llm=ScriptedReviewerLLM(report))
    assert result["status"] == "awaiting_reviewer_approval"
    assert result["review_report"]["verdict"] == "approve"
    assert all(result["review_report"]["acceptance_criteria_results"].values())
    assert all(result["review_report"]["manifest_results"].values())


def test_missing_required_file(workspace: tuple[str, Path]) -> None:
    """Missing manifest files block approval."""
    _, root = workspace
    _seed_complete_project(root / "candidate")
    (root / "candidate" / "README.md").unlink()
    plan = _valid_plan_dict()
    report = ReviewReport(
        verdict="approve",  # model incorrectly approves
        acceptance_criteria_results=_all_criteria_true(plan),
        manifest_results=_manifest_true(plan),
        reviewed_files=["src/demo_app/main.py"],
        findings=[],
        residual_risks=[],
        summary="Looks fine.",
    )
    result = reviewer_node(_base_state(root, plan), llm=ScriptedReviewerLLM(report))
    assert result["review_report"]["verdict"] == "request_changes"
    assert result["review_report"]["manifest_results"]["README.md"] is False
    assert any(
        finding["finding_id"] == "MISSING_MANIFEST_FILES"
        for finding in result["review_report"]["findings"]
    )


def test_failed_acceptance_criterion(workspace: tuple[str, Path]) -> None:
    """Failed acceptance criteria prevent approval."""
    _, root = workspace
    _seed_complete_project(root / "candidate")
    plan = _valid_plan_dict()
    ac = _all_criteria_true(plan)
    ac["task:T2:0"] = False
    report = ReviewReport(
        verdict="request_changes",
        acceptance_criteria_results=ac,
        manifest_results=_manifest_true(plan),
        reviewed_files=["src/demo_app/main.py"],
        findings=[
            ReviewFinding(
                finding_id="AC-T2-0",
                severity="major",
                category="requirements",
                file="src/demo_app/main.py",
                line=1,
                description="main.py does not expose a callable entrypoint named main.",
                evidence="No def main( symbol found during review.",
                recommendation="Add a main() function and cover it with a unit test.",
            )
        ],
        residual_risks=[],
        summary="Acceptance criterion for main() failed.",
    )
    result = reviewer_node(_base_state(root, plan), llm=ScriptedReviewerLLM(report))
    assert result["review_report"]["verdict"] == "request_changes"
    assert result["review_report"]["acceptance_criteria_results"]["task:T2:0"] is False


def test_blocking_security_defect(workspace: tuple[str, Path]) -> None:
    """Blocking security findings force request_changes even if model approves."""
    _, root = workspace
    _seed_complete_project(root / "candidate")
    (root / "candidate" / "src" / "demo_app" / "main.py").write_text(
        "API_KEY = 'sk-live-hardcoded'\n\ndef main():\n    return 0\n",
        encoding="utf-8",
    )
    plan = _valid_plan_dict()
    report = ReviewReport(
        verdict="approve",
        acceptance_criteria_results=_all_criteria_true(plan),
        manifest_results=_manifest_true(plan),
        reviewed_files=["src/demo_app/main.py"],
        findings=[
            ReviewFinding(
                finding_id="SEC-HARDCODED-KEY",
                severity="blocking",
                category="security",
                file="src/demo_app/main.py",
                line=1,
                description=(
                    "API key is hardcoded in source instead of using an environment "
                    "variable."
                ),
                evidence="API_KEY = 'sk-live-hardcoded' appears in main.py line 1.",
                recommendation=(
                    "Load the secret from os.environ and add a test that rejects "
                    "missing credentials."
                ),
            )
        ],
        residual_risks=["Secret material may already be leaked in history."],
        summary="Hardcoded credential found.",
    )
    result = reviewer_node(_base_state(root, plan), llm=ScriptedReviewerLLM(report))
    assert result["review_report"]["verdict"] == "request_changes"
    assert result["review_report"]["findings"][0]["severity"] == "blocking"


def test_major_correctness_defect(workspace: tuple[str, Path]) -> None:
    """Major correctness defects prevent approve."""
    _, root = workspace
    _seed_complete_project(root / "candidate")
    plan = _valid_plan_dict()
    report = ReviewReport(
        verdict="approve",
        acceptance_criteria_results=_all_criteria_true(plan),
        manifest_results=_manifest_true(plan),
        reviewed_files=["src/demo_app/main.py"],
        findings=[
            ReviewFinding(
                finding_id="CORR-MAIN-RETURN",
                severity="major",
                category="correctness",
                file="src/demo_app/main.py",
                line=2,
                description="main() returns 1 on success, contradicting the exit-code=0 criterion.",
                evidence="return 1 on line 2 while stories require exit code 0.",
                recommendation="Return 0 on success and update tests accordingly.",
            )
        ],
        residual_risks=[],
        summary="Correctness defect in main().",
    )
    result = reviewer_node(_base_state(root, plan), llm=ScriptedReviewerLLM(report))
    assert result["review_report"]["verdict"] == "request_changes"


def test_minor_documentation_defect(workspace: tuple[str, Path]) -> None:
    """Minor documentation findings alone still allow approve."""
    _, root = workspace
    _seed_complete_project(root / "candidate")
    plan = _valid_plan_dict()
    report = ReviewReport(
        verdict="approve",
        acceptance_criteria_results=_all_criteria_true(plan),
        manifest_results=_manifest_true(plan),
        reviewed_files=["README.md"],
        findings=[
            ReviewFinding(
                finding_id="DOC-TYPO",
                severity="minor",
                category="documentation",
                file="README.md",
                line=1,
                description="README title uses inconsistent project capitalization.",
                evidence="Title is '# Demo' while project_name is demo_app.",
                recommendation="Align the README title with project_name=demo_app.",
            )
        ],
        residual_risks=[],
        summary="Approved with a minor documentation nit.",
    )
    result = reviewer_node(_base_state(root, plan), llm=ScriptedReviewerLLM(report))
    assert result["review_report"]["verdict"] == "approve"
    assert result["review_report"]["findings"][0]["severity"] == "minor"


def test_verification_failure(workspace: tuple[str, Path]) -> None:
    """Failed verification cannot yield approve."""
    _, root = workspace
    _seed_complete_project(root / "candidate")
    plan = _valid_plan_dict()
    report = ReviewReport(
        verdict="approve",
        acceptance_criteria_results=_all_criteria_true(plan),
        manifest_results=_manifest_true(plan),
        reviewed_files=["tests/test_main.py"],
        findings=[],
        residual_risks=[],
        summary="Code looks fine.",
    )
    state = _base_state(
        root,
        plan,
        verification_report={
            "passed": False,
            "overall_status": "failed",
            "commands": [
                {
                    "name": "pytest",
                    "exit_code": 1,
                    "stderr": "AssertionError",
                }
            ],
        },
    )
    result = reviewer_node(state, llm=ScriptedReviewerLLM(report))
    assert result["review_report"]["verdict"] == "request_changes"
    assert any(
        finding["finding_id"] == "VERIFICATION_FAILED"
        for finding in result["review_report"]["findings"]
    )


def test_false_coder_resolution_claim(workspace: tuple[str, Path]) -> None:
    """Unverified coder resolution claims surface as review findings."""
    _, root = workspace
    _seed_complete_project(root / "candidate")
    plan = _valid_plan_dict()
    report = ReviewReport(
        verdict="request_changes",
        acceptance_criteria_results=_all_criteria_true(plan),
        manifest_results=_manifest_true(plan),
        reviewed_files=["src/demo_app/main.py"],
        findings=[
            ReviewFinding(
                finding_id="FALSE-RESOLUTION-REV-1",
                severity="major",
                category="correctness",
                file="src/demo_app/main.py",
                line=1,
                description=(
                    "Coder claimed REV-1 was fixed by adding input validation, but "
                    "main() still accepts arbitrary input without checks."
                ),
                evidence=(
                    "feedback_resolutions['REV-1'] claims validation was added, but "
                    "src/demo_app/main.py contains only `def main(): return 0`."
                ),
                recommendation=(
                    "Implement real argument validation in main() and add a failing "
                    "test for invalid input before marking REV-1 resolved."
                ),
            )
        ],
        residual_risks=[],
        summary="Coder resolution claim for REV-1 is not supported by the code.",
    )
    state = _base_state(
        root,
        plan,
        coder_result={
            "summary": "fixed review findings",
            "created_files": [],
            "modified_files": ["src/demo_app/main.py"],
            "deleted_files": [],
            "unresolved_issues": [],
            "feedback_resolutions": {
                "REV-1": "Added argv validation in main().",
            },
            "manifest_compliance": _manifest_true(plan),
        },
        previous_review_report={
            "verdict": "request_changes",
            "findings": [{"finding_id": "REV-1", "severity": "major"}],
        },
    )
    result = reviewer_node(state, llm=ScriptedReviewerLLM(report))
    assert result["review_report"]["verdict"] == "request_changes"
    assert result["review_report"]["findings"][0]["finding_id"] == (
        "FALSE-RESOLUTION-REV-1"
    )


def test_request_changes_verdict(workspace: tuple[str, Path]) -> None:
    """Explicit request_changes verdict is preserved for implementation issues."""
    _, root = workspace
    _seed_complete_project(root / "candidate")
    plan = _valid_plan_dict()
    report = ReviewReport(
        verdict="request_changes",
        acceptance_criteria_results=_all_criteria_true(plan),
        manifest_results=_manifest_true(plan),
        reviewed_files=["tests/test_main.py"],
        findings=[
            ReviewFinding(
                finding_id="TEST-WEAK",
                severity="major",
                category="testing",
                file="tests/test_main.py",
                line=3,
                description="Test suite asserts only truthiness, not the exit-code contract.",
                evidence="test_main uses `assert main()` instead of `assert main() == 0`.",
                recommendation="Assert equality with 0 and cover an error path.",
            )
        ],
        residual_risks=[],
        summary="Implementation changes required for test quality.",
    )
    result = reviewer_node(_base_state(root, plan), llm=ScriptedReviewerLLM(report))
    assert result["review_report"]["verdict"] == "request_changes"


def test_replan_verdict(workspace: tuple[str, Path]) -> None:
    """Replan is used when architecture/task decomposition is insufficient."""
    _, root = workspace
    _seed_complete_project(root / "candidate")
    plan = _valid_plan_dict()
    report = ReviewReport(
        verdict="replan",
        acceptance_criteria_results=_all_criteria_true(plan),
        manifest_results=_manifest_true(plan),
        reviewed_files=["src/demo_app/main.py"],
        findings=[
            ReviewFinding(
                finding_id="ARCH-MISMATCH",
                severity="blocking",
                category="architecture",
                file=None,
                line=None,
                description=(
                    "The approved modular-monolith plan cannot satisfy the request for "
                    "an asynchronous multi-tenant event pipeline."
                ),
                evidence=(
                    "User request requires multi-tenant async workers, but the plan "
                    "selects a single-process CLI without queues or tenancy boundaries."
                ),
                recommendation=(
                    "Replan with an architecture that includes tenancy isolation and "
                    "an explicit async processing component justified by the request."
                ),
            )
        ],
        residual_risks=["Continuing to iterate on the current plan wastes iterations."],
        summary="Architecture is contradictory; replan required.",
    )
    result = reviewer_node(_base_state(root, plan), llm=ScriptedReviewerLLM(report))
    assert result["review_report"]["verdict"] == "replan"


def test_no_write_tools_exposed(workspace: tuple[str, Path]) -> None:
    """Reviewer tools expose only read-only operations."""
    _, root = workspace
    tools = ReadOnlyWorkspaceTools(
        root / "candidate",
        verification_report={"passed": True},
    )
    names = set(tools.tool_names())
    assert names == set(READ_ONLY_TOOL_NAMES)
    assert names.isdisjoint(FORBIDDEN_MUTATION_TOOL_NAMES)

    before = "untouched\n"
    target = root / "candidate" / "marker.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(before, encoding="utf-8")

    with pytest.raises(ValueError, match="mutation tool"):
        tools.invoke_tool("write_file", {"path": "marker.txt", "content": "mutated\n"})
    with pytest.raises(ValueError, match="mutation tool"):
        tools.invoke_tool("delete_file", {"path": "marker.txt"})

    assert target.read_text(encoding="utf-8") == before

    # Running the node with a tool loop that only lists files must not mutate.
    plan = _valid_plan_dict()
    _seed_complete_project(root / "candidate")
    report = ReviewReport(
        verdict="approve",
        acceptance_criteria_results=_all_criteria_true(plan),
        manifest_results=_manifest_true(plan),
        reviewed_files=[],
        findings=[],
        residual_risks=[],
        summary="ok",
    )
    llm = ScriptedReviewerLLM(
        report,
        tool_script=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "list_files",
                        "args": {},
                        "id": "t1",
                        "type": "tool_call",
                    }
                ],
            )
        ],
    )
    reviewer_node(_base_state(root, plan), llm=llm)
    assert target.read_text(encoding="utf-8") == before
    assert "write_file" not in names


def test_missing_workspace_raises(tmp_path: Path) -> None:
    """workspace_path is required."""
    with pytest.raises(ValueError, match="workspace_path"):
        reviewer_node(
            {
                "user_request": "x",
                "plan": _valid_plan_dict(),
                "generated_files": [],
                "verification_report": {},
                "coder_result": {},
            },
            llm=ScriptedReviewerLLM(
                ReviewReport(
                    verdict="request_changes",
                    summary="n/a",
                )
            ),
        )
