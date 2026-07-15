"""Unit tests for the coder node (mocked LLM; temp workspaces)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from codegen_workflow.nodes.coder import (
    coder_node,
    extract_findings,
    run_tool_loop,
)
from codegen_workflow.tools.workspace import WorkspaceFileTools
from codegen_workflow.workspace import create_workspace


def _valid_plan_dict(**overrides: Any) -> dict[str, Any]:
    """Minimal valid plan accepted by the coder plan validator."""
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


class ScriptedLLM:
    """Deterministic tool-calling stand-in for ChatOpenAI."""

    def __init__(self, script: list[AIMessage]) -> None:
        self.script = list(script)
        self.calls = 0

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedLLM:
        return self

    def invoke(
        self, messages: Any, config: Any | None = None, **kwargs: Any
    ) -> AIMessage:
        if self.calls >= len(self.script):
            return AIMessage(content="done")
        message = self.script[self.calls]
        self.calls += 1
        return message


def _tool_call(name: str, args: dict[str, Any], call_id: str) -> dict[str, Any]:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def _full_generation_script() -> list[AIMessage]:
    """Script that creates all mandatory manifest files then stops."""
    return [
        AIMessage(
            content="",
            tool_calls=[
                _tool_call(
                    "write_file",
                    {
                        "path": "pyproject.toml",
                        "content": "[project]\nname='demo_app'\n",
                    },
                    "c1",
                )
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                _tool_call(
                    "write_file",
                    {
                        "path": "src/demo_app/main.py",
                        "content": "def main():\n    return 0\n",
                    },
                    "c2",
                )
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                _tool_call(
                    "write_file",
                    {
                        "path": "tests/test_main.py",
                        "content": "from demo_app.main import main\n\ndef test_main():\n    assert main() == 0\n",
                    },
                    "c3",
                )
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                _tool_call(
                    "write_file",
                    {
                        "path": "README.md",
                        "content": "# demo_app\n\n## Install\npip install -e .\n\n## Run\npython -m demo_app\n",
                    },
                    "c4",
                )
            ],
        ),
        AIMessage(content="Created configuration, source, tests, and docs."),
    ]


@pytest.fixture
def workspace(tmp_path: Path) -> tuple[str, Path]:
    """Create an isolated workflow workspace under tmp_path."""
    return create_workspace(base_dir=tmp_path / "workspaces")


def test_create_files_from_empty_workspace(workspace: tuple[str, Path]) -> None:
    """Coder creates planned files in an empty candidate workspace."""
    _, root = workspace
    llm = ScriptedLLM(_full_generation_script())
    result = coder_node(
        {
            "user_request": "Build demo_app",
            "plan": _valid_plan_dict(),
            "workspace_path": str(root),
            "iteration": 0,
            "feedback_history": [],
            "verification_report": {},
            "review_report": {},
            "max_iterations": 4,
        },
        llm=llm,
    )
    assert result["status"] == "verifying"
    assert result["iteration"] == 1
    assert "README.md" in result["generated_files"]
    assert "src/demo_app/main.py" in result["generated_files"]
    assert (root / "candidate" / "src" / "demo_app" / "main.py").is_file()
    assert result["coder_result"]["manifest_compliance"]["README.md"] is True


def test_create_nested_directories_via_coder(workspace: tuple[str, Path]) -> None:
    """Nested package paths are created when writing source files."""
    _, root = workspace
    llm = ScriptedLLM(_full_generation_script())
    coder_node(
        {
            "user_request": "Build demo_app",
            "plan": _valid_plan_dict(),
            "workspace_path": str(root),
            "iteration": 0,
            "feedback_history": [],
            "max_iterations": 4,
        },
        llm=llm,
    )
    assert (root / "candidate" / "src" / "demo_app").is_dir()


def test_update_existing_file(workspace: tuple[str, Path]) -> None:
    """Revision iteration modifies an existing file in place."""
    _, root = workspace
    candidate = root / "candidate"
    (candidate / "pyproject.toml").write_text(
        "[project]\nname='demo_app'\n", encoding="utf-8"
    )
    (candidate / "src" / "demo_app").mkdir(parents=True)
    (candidate / "src" / "demo_app" / "main.py").write_text(
        "def main():\n    return 1\n", encoding="utf-8"
    )
    (candidate / "tests").mkdir()
    (candidate / "tests" / "test_main.py").write_text(
        "def test_main():\n    pass\n", encoding="utf-8"
    )
    (candidate / "README.md").write_text("# demo\n", encoding="utf-8")

    llm = ScriptedLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call(
                        "write_file",
                        {
                            "path": "src/demo_app/main.py",
                            "content": "def main():\n    return 0\n",
                        },
                        "u1",
                    )
                ],
            ),
            AIMessage(content="Fixed main return value."),
        ]
    )
    result = coder_node(
        {
            "user_request": "Build demo_app",
            "plan": _valid_plan_dict(),
            "workspace_path": str(root),
            "iteration": 1,
            "feedback_history": [],
            "max_iterations": 4,
        },
        llm=llm,
    )
    assert result["iteration"] == 2
    assert "src/demo_app/main.py" in result["coder_result"]["modified_files"]
    assert (candidate / "src" / "demo_app" / "main.py").read_text(encoding="utf-8") == (
        "def main():\n    return 0\n"
    )


def test_delete_explicitly_obsolete_file(workspace: tuple[str, Path]) -> None:
    """Coder can delete a file when the scripted model requests it."""
    _, root = workspace
    obsolete = root / "candidate" / "obsolete.py"
    obsolete.write_text("old\n", encoding="utf-8")
    llm = ScriptedLLM(
        [
            AIMessage(
                content="",
                tool_calls=[_tool_call("delete_file", {"path": "obsolete.py"}, "d1")],
            ),
            *_full_generation_script(),
        ]
    )
    result = coder_node(
        {
            "user_request": "Build demo_app",
            "plan": _valid_plan_dict(),
            "workspace_path": str(root),
            "iteration": 0,
            "feedback_history": [],
            "max_iterations": 4,
        },
        llm=llm,
    )
    assert "obsolete.py" in result["coder_result"]["deleted_files"]
    assert not obsolete.exists()


def test_detect_missing_manifest_files(workspace: tuple[str, Path]) -> None:
    """Incomplete generation reports missing manifest paths."""
    _, root = workspace
    llm = ScriptedLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call(
                        "write_file",
                        {"path": "README.md", "content": "# only docs\n"},
                        "m1",
                    )
                ],
            ),
            AIMessage(content="Only wrote README."),
        ]
    )
    result = coder_node(
        {
            "user_request": "Build demo_app",
            "plan": _valid_plan_dict(),
            "workspace_path": str(root),
            "iteration": 0,
            "feedback_history": [],
            "max_iterations": 4,
        },
        llm=llm,
    )
    compliance = result["coder_result"]["manifest_compliance"]
    assert compliance["README.md"] is True
    assert compliance["src/demo_app/main.py"] is False
    assert any(
        "missing_manifest_files" in issue
        for issue in result["coder_result"]["unresolved_issues"]
    )


def test_process_human_feedback(workspace: tuple[str, Path]) -> None:
    """Human feedback findings are mapped to feedback_resolutions."""
    _, root = workspace
    # Seed existing complete project.
    for message in _full_generation_script()[:-1]:
        call = message.tool_calls[0]
        path = call["args"]["path"]
        content = call["args"]["content"]
        target = root / "candidate" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    finding_id = "coder_1_0"
    llm = ScriptedLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call(
                        "write_file",
                        {
                            "path": "README.md",
                            "content": "# demo_app\n\nVerbose mode documented.\n",
                        },
                        "h1",
                    )
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call(
                        "resolve_feedback",
                        {
                            "finding_id": finding_id,
                            "resolution": "Documented verbose mode in README.md",
                        },
                        "h2",
                    )
                ],
            ),
            AIMessage(content="Addressed human feedback."),
        ]
    )
    result = coder_node(
        {
            "user_request": "Build demo_app",
            "plan": _valid_plan_dict(),
            "workspace_path": str(root),
            "iteration": 1,
            "max_iterations": 4,
            "feedback_history": [
                {
                    "gate": "coder",
                    "iteration": 1,
                    "decision": "request_changes",
                    "feedback": "Document the --verbose flag.",
                }
            ],
            "verification_report": {"passed": True},
            "review_report": {},
        },
        llm=llm,
    )
    assert result["coder_result"]["feedback_resolutions"][finding_id].startswith(
        "Documented"
    )


def test_process_reviewer_findings(workspace: tuple[str, Path]) -> None:
    """Reviewer findings are extractable and resolvable by id."""
    findings = extract_findings(
        {
            "feedback_history": [],
            "verification_report": {"passed": True},
            "review_report": {
                "findings": [
                    {
                        "id": "REV-1",
                        "description": "Add input validation to main().",
                    }
                ]
            },
        }
    )
    assert findings == [("REV-1", "Add input validation to main().")]

    _, root = workspace
    for message in _full_generation_script()[:-1]:
        call = message.tool_calls[0]
        target = root / "candidate" / call["args"]["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(call["args"]["content"], encoding="utf-8")

    llm = ScriptedLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call(
                        "write_file",
                        {
                            "path": "src/demo_app/main.py",
                            "content": (
                                "def main(argv=None):\n"
                                "    if argv is None:\n"
                                "        argv = []\n"
                                "    return 0\n"
                            ),
                        },
                        "r1",
                    )
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call(
                        "resolve_feedback",
                        {
                            "finding_id": "REV-1",
                            "resolution": "Added argv validation in main().",
                        },
                        "r2",
                    )
                ],
            ),
            AIMessage(content="Resolved reviewer finding."),
        ]
    )
    result = coder_node(
        {
            "user_request": "Build demo_app",
            "plan": _valid_plan_dict(),
            "workspace_path": str(root),
            "iteration": 1,
            "max_iterations": 4,
            "feedback_history": [],
            "verification_report": {"passed": True},
            "review_report": {
                "findings": [
                    {
                        "id": "REV-1",
                        "description": "Add input validation to main().",
                    }
                ]
            },
        },
        llm=llm,
    )
    assert "REV-1" in result["coder_result"]["feedback_resolutions"]


def test_process_verification_failures(workspace: tuple[str, Path]) -> None:
    """Failed verification commands become findings for the coder."""
    findings = extract_findings(
        {
            "feedback_history": [],
            "review_report": {},
            "verification_report": {
                "passed": False,
                "overall_status": "failed",
                "commands": [
                    {
                        "name": "pytest",
                        "exit_code": 1,
                        "stderr": "AssertionError: expected 0",
                    }
                ],
            },
        }
    )
    assert findings[0][0] == "verification_pytest"

    _, root = workspace
    for message in _full_generation_script()[:-1]:
        call = message.tool_calls[0]
        target = root / "candidate" / call["args"]["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(call["args"]["content"], encoding="utf-8")

    llm = ScriptedLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call(
                        "write_file",
                        {
                            "path": "tests/test_main.py",
                            "content": (
                                "from demo_app.main import main\n\n"
                                "def test_main():\n    assert main() == 0\n"
                            ),
                        },
                        "v1",
                    )
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call(
                        "resolve_feedback",
                        {
                            "finding_id": "verification_pytest",
                            "resolution": "Corrected assertions to expect 0.",
                        },
                        "v2",
                    )
                ],
            ),
            AIMessage(content="Fixed failing tests."),
        ]
    )
    result = coder_node(
        {
            "user_request": "Build demo_app",
            "plan": _valid_plan_dict(),
            "workspace_path": str(root),
            "iteration": 1,
            "max_iterations": 4,
            "feedback_history": [],
            "review_report": {},
            "verification_report": {
                "passed": False,
                "overall_status": "failed",
                "commands": [
                    {
                        "name": "pytest",
                        "exit_code": 1,
                        "stderr": "AssertionError: expected 0",
                    }
                ],
            },
        },
        llm=llm,
    )
    assert "verification_pytest" in result["coder_result"]["feedback_resolutions"]


def test_maintain_file_hashes(workspace: tuple[str, Path]) -> None:
    """Coder state update includes SHA-256 hashes for generated files."""
    _, root = workspace
    llm = ScriptedLLM(_full_generation_script())
    result = coder_node(
        {
            "user_request": "Build demo_app",
            "plan": _valid_plan_dict(),
            "workspace_path": str(root),
            "iteration": 0,
            "feedback_history": [],
            "max_iterations": 4,
        },
        llm=llm,
    )
    assert set(result["file_hashes"]) == set(result["generated_files"])
    assert all(len(digest) == 64 for digest in result["file_hashes"].values())


def test_bounded_model_tool_execution(workspace: tuple[str, Path]) -> None:
    """Exceeding the tool-call budget yields a typed coder failure."""
    _, root = workspace
    endless = [
        AIMessage(
            content="",
            tool_calls=[
                _tool_call(
                    "list_files",
                    {},
                    f"loop_{index}",
                )
            ],
        )
        for index in range(10)
    ]
    llm = ScriptedLLM(endless)
    result = coder_node(
        {
            "user_request": "Build demo_app",
            "plan": _valid_plan_dict(),
            "workspace_path": str(root),
            "iteration": 0,
            "feedback_history": [],
            "max_iterations": 4,
        },
        llm=llm,
        max_tool_calls=3,
    )
    assert result["status"] == "coder_failed"
    assert result["errors"][0]["type"] == "tool_limit"
    assert result["iteration"] == 1


def test_run_tool_loop_direct(tmp_path: Path) -> None:
    """Bounded loop executes tool calls against WorkspaceFileTools."""
    tools = WorkspaceFileTools(tmp_path / "candidate")
    llm = ScriptedLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call(
                        "write_file",
                        {"path": "a.py", "content": "x=1\n"},
                        "t1",
                    )
                ],
            ),
            AIMessage(content="wrote a.py"),
        ]
    )
    messages, count, summary = run_tool_loop(llm, tools, [], max_tool_calls=5)
    assert count == 1
    assert summary == "wrote a.py"
    assert tools.read_file("a.py") == "x=1\n"
    assert isinstance(messages[-1], AIMessage)


def test_missing_plan_fails(workspace: tuple[str, Path]) -> None:
    """Coder fails fast when plan is absent."""
    _, root = workspace
    result = coder_node(
        {
            "user_request": "Build something",
            "plan": {},
            "workspace_path": str(root),
            "iteration": 0,
        },
        llm=ScriptedLLM([]),
    )
    assert result["status"] == "coder_failed"
    assert result["errors"][0]["type"] == "invalid_input"
