"""Unit tests for the planner node (mocked LLM; no live API calls)."""

from __future__ import annotations

from copy import deepcopy
from typing import Any
from unittest.mock import MagicMock

from codegen_workflow.nodes.planner import (
    MAX_USER_REQUEST_CHARS,
    build_planner_prompt,
    planner_node,
)
from codegen_workflow.schemas.plan import ProjectPlan


def _valid_plan_dict(**overrides: Any) -> dict[str, Any]:
    """Minimal valid plan shared with schema tests."""
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


def _api_plan() -> ProjectPlan:
    """Return a valid plan shaped like an HTTP API project."""
    payload = _valid_plan_dict(
        project_name="items_api",
        objective="Expose a small REST API for managing items.",
        framework="fastapi",
        architecture_pattern="layered API",
        dependencies=["fastapi", "uvicorn", "pytest", "httpx"],
        run_command="uvicorn items_api.main:app --reload",
        validation_commands=["pytest -q"],
    )
    payload["file_manifest"][1]["path"] = "src/items_api/main.py"
    payload["tasks"][1]["files"] = ["src/items_api/main.py"]
    payload["file_manifest"][2]["depends_on"] = ["src/items_api/main.py"]
    payload["tasks"][2]["files"] = ["tests/test_main.py"]
    return ProjectPlan.model_validate(payload)


def _cli_plan() -> ProjectPlan:
    """Return a valid plan shaped like a command-line application."""
    payload = _valid_plan_dict(
        project_name="greeter_cli",
        objective="Provide a CLI that greets a user by name.",
        framework=None,
        architecture_pattern="single-package CLI",
        dependencies=["pytest"],
        run_command="python -m greeter_cli",
        validation_commands=["pytest -q"],
    )
    payload["stories"][0]["description"] = (
        "As a terminal user, I want to pass my name, so that I receive a greeting."
    )
    payload["file_manifest"][1]["path"] = "src/greeter_cli/main.py"
    payload["tasks"][1]["files"] = ["src/greeter_cli/main.py"]
    payload["file_manifest"][2]["depends_on"] = ["src/greeter_cli/main.py"]
    return ProjectPlan.model_validate(payload)


def test_valid_api_generation_request() -> None:
    """Planner accepts an API requirement and returns a validated plan."""
    llm = MagicMock()
    llm.invoke.return_value = _api_plan()

    result = planner_node(
        {
            "user_request": (
                "Build a FastAPI service with a /health endpoint and a CRUD "
                "/items API backed by in-memory storage."
            ),
            "planner_feedback": [],
        },
        llm=llm,
    )

    assert result["status"] == "coding"
    assert result["planner_errors"] == []
    assert result["plan"]["project_name"] == "items_api"
    assert result["plan"]["framework"] == "fastapi"
    assert result["plan"]["epics"]
    assert result["plan"]["stories"]
    assert result["plan"]["tasks"]
    assert result["plan"]["file_manifest"]
    llm.invoke.assert_called_once()


def test_valid_cli_application_request() -> None:
    """Planner accepts a CLI requirement and returns a validated plan."""
    llm = MagicMock()
    llm.invoke.return_value = _cli_plan()

    result = planner_node(
        {
            "user_request": "Create a Python CLI that greets the user by name.",
            "planner_feedback": [],
        },
        llm=llm,
    )

    assert result["status"] == "coding"
    assert result["plan"]["project_name"] == "greeter_cli"
    assert result["plan"]["run_command"] == "python -m greeter_cli"
    assert any(task["task_type"] == "source" for task in result["plan"]["tasks"])
    assert any(task["task_type"] == "test" for task in result["plan"]["tasks"])


def test_empty_user_input() -> None:
    """Empty or whitespace-only user_request fails without calling the model."""
    llm = MagicMock()
    result = planner_node({"user_request": "   ", "planner_feedback": []}, llm=llm)
    assert result["status"] == "planner_failed"
    assert result["plan"] == {}
    assert result["planner_errors"][0]["type"] == "invalid_input"
    llm.invoke.assert_not_called()

    result_missing = planner_node({"planner_feedback": []}, llm=llm)
    assert result_missing["planner_errors"][0]["type"] == "invalid_input"


def test_user_request_exceeds_limit() -> None:
    """Overlong user_request is rejected as invalid input."""
    llm = MagicMock()
    result = planner_node(
        {"user_request": "x" * (MAX_USER_REQUEST_CHARS + 1), "planner_feedback": []},
        llm=llm,
    )
    assert result["status"] == "planner_failed"
    assert result["planner_errors"][0]["type"] == "invalid_input"
    llm.invoke.assert_not_called()


def test_duplicate_task_ids() -> None:
    """Plans with duplicate task ids return a validation error state."""
    payload = _valid_plan_dict()
    payload["tasks"][1] = deepcopy(payload["tasks"][0])
    llm = MagicMock()
    llm.invoke.return_value = ProjectPlan.model_validate(payload)

    result = planner_node(
        {"user_request": "Build a demo app", "planner_feedback": []},
        llm=llm,
    )
    assert result["status"] == "planner_failed"
    assert result["planner_errors"][0]["type"] == "validation_error"
    assert any("unique" in detail for detail in result["planner_errors"][0]["details"])


def test_missing_task_dependencies() -> None:
    """Missing dependency references surface as validation errors."""
    payload = _valid_plan_dict()
    payload["tasks"][1]["dependencies"] = ["T_DOES_NOT_EXIST"]
    llm = MagicMock()
    llm.invoke.return_value = ProjectPlan.model_validate(payload)

    result = planner_node(
        {"user_request": "Build a demo app", "planner_feedback": []},
        llm=llm,
    )
    assert result["status"] == "planner_failed"
    assert any(
        "missing task" in detail for detail in result["planner_errors"][0]["details"]
    )


def test_cyclic_task_dependencies() -> None:
    """Cyclic dependencies are rejected by the planner node."""
    payload = _valid_plan_dict()
    payload["tasks"][0]["dependencies"] = ["T3"]
    payload["tasks"][2]["dependencies"] = ["T1"]
    llm = MagicMock()
    llm.invoke.return_value = ProjectPlan.model_validate(payload)

    result = planner_node(
        {"user_request": "Build a demo app", "planner_feedback": []},
        llm=llm,
    )
    assert result["status"] == "planner_failed"
    assert any("cycle" in detail for detail in result["planner_errors"][0]["details"])


def test_unsafe_file_paths() -> None:
    """Unsafe file paths cause planner validation failure."""
    payload = _valid_plan_dict()
    payload["file_manifest"][1]["path"] = "../../etc/passwd"
    llm = MagicMock()
    llm.invoke.return_value = ProjectPlan.model_validate(payload)

    result = planner_node(
        {"user_request": "Build a demo app", "planner_feedback": []},
        llm=llm,
    )
    assert result["status"] == "planner_failed"
    assert any(".." in detail for detail in result["planner_errors"][0]["details"])


def test_plan_without_tests() -> None:
    """Plans lacking automated tests are accepted (tests are optional)."""
    payload = _valid_plan_dict()
    payload["tasks"] = [
        task for task in payload["tasks"] if task["task_type"] != "test"
    ]
    payload["file_manifest"] = [
        spec for spec in payload["file_manifest"] if spec["file_type"] != "test"
    ]
    llm = MagicMock()
    llm.invoke.return_value = ProjectPlan.model_validate(payload)

    result = planner_node(
        {"user_request": "Build a demo app", "planner_feedback": []},
        llm=llm,
    )
    assert result["status"] == "coding"
    assert result.get("plan") is not None


def test_model_returns_invalid_structured_output() -> None:
    """Invalid model payloads map to invalid_structured_output errors."""
    llm = MagicMock()
    llm.invoke.return_value = {"not": "a valid plan"}

    result = planner_node(
        {"user_request": "Build a demo app", "planner_feedback": []},
        llm=llm,
    )
    assert result["status"] == "planner_failed"
    assert result["planner_errors"][0]["type"] == "invalid_structured_output"
    assert llm.invoke.call_count == 3


def test_model_timeout() -> None:
    """TimeoutError maps to a typed model_timeout planner error."""
    llm = MagicMock()
    llm.invoke.side_effect = TimeoutError("deadline exceeded")

    result = planner_node(
        {"user_request": "Build a demo app", "planner_feedback": []},
        llm=llm,
    )
    assert result["status"] == "planner_failed"
    assert result["planner_errors"][0]["type"] == "model_timeout"


def test_planner_feedback_incorporated_during_replanning() -> None:
    """Replanning prompts include prior planner_feedback items."""
    llm = MagicMock()
    llm.invoke.return_value = _cli_plan()
    feedback = [
        "Prefer argparse over click.",
        "Add a --verbose flag to the CLI.",
    ]

    result = planner_node(
        {
            "user_request": "Create a Python CLI that greets the user by name.",
            "planner_feedback": feedback,
        },
        llm=llm,
    )

    assert result["status"] == "coding"
    messages = llm.invoke.call_args.args[0]
    human_contents = [
        message.content
        for message in messages
        if getattr(message, "type", "") == "human"
    ]
    prompt_text = "\n".join(human_contents)
    assert "Prefer argparse over click." in prompt_text
    assert "Add a --verbose flag to the CLI." in prompt_text
    assert "Planner feedback from prior review" in prompt_text


def test_build_planner_prompt_without_feedback() -> None:
    """Prompt construction marks absent feedback explicitly."""
    prompt = build_planner_prompt("Build a todo API")
    assert "Build a todo API" in prompt
    assert "None." in prompt


def test_validation_error_from_pydantic_on_bad_dict() -> None:
    """Direct ValidationError paths are covered via a broken coerce target."""
    llm = MagicMock()

    # Return something that _coerce_plan treats as a dict but fails ProjectPlan.
    class BrokenDict(dict):
        pass

    llm.invoke.return_value = BrokenDict(project_name=123)

    result = planner_node(
        {"user_request": "Build a demo app", "planner_feedback": []},
        llm=llm,
    )
    assert result["status"] == "planner_failed"
    assert result["planner_errors"][0]["type"] == "invalid_structured_output"


def test_initial_planning_has_no_previous_plan_or_diff() -> None:
    """Initial planning does not populate revision fields."""
    llm = MagicMock()
    llm.invoke.return_value = _cli_plan()
    result = planner_node(
        {"user_request": "Create a Python CLI that greets the user by name."},
        llm=llm,
    )
    assert result["status"] == "coding"
    assert result.get("previous_plan") == {}
    assert result.get("plan_diff") == {}


def test_revision_planning_preserves_previous_and_computes_diff() -> None:
    """Revision mode stores previous_plan and a deterministic plan_diff."""
    current = _valid_plan_dict()
    revised_dict = _valid_plan_dict()
    revised_dict["file_manifest"].append(
        {
            "path": "src/demo_app/auth.py",
            "purpose": "Authentication helpers.",
            "file_type": "source",
            "requirements": ["Defines authenticate()."],
            "depends_on": ["src/demo_app/main.py"],
        }
    )
    llm = MagicMock()
    llm.invoke.return_value = ProjectPlan.model_validate(revised_dict)

    result = planner_node(
        {
            "user_request": "Create a Python CLI that greets the user by name.",
            "plan": current,
            "change_request": {
                "feedback": "Add JWT authentication",
                "source_gate": "coder",
                "iteration": 1,
            },
            "generated_files": ["src/demo_app/main.py", "README.md"],
            "verification_report": {"passed": False},
            "review_report": {},
            "feedback_history": [],
        },
        llm=llm,
    )

    assert result["status"] == "coding"
    assert result["previous_plan"] == current
    assert result["plan"]["project_name"] == revised_dict["project_name"]
    assert "src/demo_app/auth.py" in result["plan_diff"]["added"]
    assert result["change_request"] == {}
    messages = llm.invoke.call_args.args[0]
    prompt_text = "\n".join(
        message.content
        for message in messages
        if getattr(message, "type", "") == "human"
    )
    assert "Revision mode" in prompt_text
    assert "Add JWT authentication" in prompt_text
    assert "Current ProjectPlan" in prompt_text


def test_invalid_revised_plan_does_not_replace_current_plan() -> None:
    """Failed revision planning keeps the authoritative plan."""
    current = _valid_plan_dict()
    llm = MagicMock()
    llm.invoke.return_value = {"project_name": "broken"}

    result = planner_node(
        {
            "user_request": "Create a Python CLI that greets the user by name.",
            "plan": current,
            "change_request": {
                "feedback": "Add auth",
                "source_gate": "coder",
                "iteration": 1,
            },
        },
        llm=llm,
    )
    assert result["status"] == "planner_failed"
    assert result["plan"] == current


def test_static_to_api_revision_prompt_requires_app_checklist() -> None:
    """Upgrading a static site to an API includes application-plan requirements."""
    from codegen_workflow.nodes.planner import build_revision_prompt

    prompt = build_revision_prompt(
        {
            "plan": {
                "project_name": "landing",
                "objective": "Static marketplace landing page.",
                "language": "HTML",
                "architecture_pattern": "static site",
                "dependencies": [],
                "file_manifest": [
                    {
                        "path": "index.html",
                        "purpose": "page",
                        "file_type": "source",
                        "requirements": ["exists"],
                        "depends_on": [],
                    },
                    {
                        "path": "README.md",
                        "purpose": "docs",
                        "file_type": "documentation",
                        "requirements": ["exists"],
                        "depends_on": [],
                    },
                ],
                "epics": [
                    {
                        "id": "E1",
                        "title": "Landing",
                        "description": "Landing page",
                        "acceptance_criteria": ["index.html exists with a header."],
                    }
                ],
                "stories": [
                    {
                        "id": "S1",
                        "epic_id": "E1",
                        "title": "Page",
                        "description": "As a visitor I want a landing page.",
                        "acceptance_criteria": ["index.html has a main section."],
                    }
                ],
                "tasks": [
                    {
                        "id": "T1",
                        "story_id": "S1",
                        "title": "HTML",
                        "description": "Write HTML",
                        "task_type": "source",
                        "dependencies": [],
                        "files": ["index.html"],
                        "acceptance_criteria": ["index.html file exists on disk."],
                    }
                ],
                "install_commands": [],
                "validation_commands": [
                    'python3 -c "from pathlib import Path; assert Path(\'index.html\').is_file()"'
                ],
                "assumptions": [],
                "framework": None,
                "run_command": None,
                "risks": [],
            },
            "change_request": {
                "feedback": "Add a FastAPI backend API for product listings",
                "source_gate": "coder",
                "iteration": 1,
            },
            "generated_files": ["index.html", "README.md"],
            "verification_report": {},
            "review_report": {},
            "feedback_history": [],
        },
        "Build a simple marketplace landing page",
    )
    assert "static site → application/API upgrade" in prompt
    assert "OPTIONAL" in prompt
    assert "Backend language/framework" in prompt
    assert "browser-open" in prompt.lower()
