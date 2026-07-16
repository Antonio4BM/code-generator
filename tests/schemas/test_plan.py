"""Tests for project-plan schemas and deterministic validation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from pydantic import ValidationError

from codegen_workflow.schemas.plan import (
    PlanValidationError,
    ProjectPlan,
    collect_plan_validation_errors,
    validate_plan,
)


def _valid_plan_dict(**overrides: Any) -> dict[str, Any]:
    """Build a minimal valid plan payload, with optional field overrides."""
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


def test_valid_plan_passes_validation() -> None:
    """A complete well-formed plan validates successfully."""
    plan = ProjectPlan.model_validate(_valid_plan_dict())
    assert validate_plan(plan) is plan
    assert collect_plan_validation_errors(plan) == []


def test_duplicate_task_ids_rejected() -> None:
    """Duplicate task identifiers fail validation."""
    payload = _valid_plan_dict()
    payload["tasks"][1] = deepcopy(payload["tasks"][0])
    payload["tasks"][1]["title"] = "Duplicate id task"
    plan = ProjectPlan.model_validate(payload)
    errors = collect_plan_validation_errors(plan)
    assert any("task ids must be unique" in error for error in errors)
    with pytest.raises(PlanValidationError):
        validate_plan(plan)


def test_missing_task_dependency_rejected() -> None:
    """Dependencies that reference unknown task ids fail validation."""
    payload = _valid_plan_dict()
    payload["tasks"][1]["dependencies"] = ["T_MISSING"]
    plan = ProjectPlan.model_validate(payload)
    errors = collect_plan_validation_errors(plan)
    assert any("depends on missing task" in error for error in errors)


def test_cyclic_task_dependencies_rejected() -> None:
    """Cyclic task dependency graphs fail validation."""
    payload = _valid_plan_dict()
    payload["tasks"][0]["dependencies"] = ["T3"]
    payload["tasks"][2]["dependencies"] = ["T1"]
    plan = ProjectPlan.model_validate(payload)
    errors = collect_plan_validation_errors(plan)
    assert any("cycle" in error for error in errors)


def test_unsafe_file_paths_rejected() -> None:
    """Absolute paths and parent traversal are rejected."""
    payload = _valid_plan_dict()
    payload["file_manifest"][1]["path"] = "../outside.py"
    plan = ProjectPlan.model_validate(payload)
    errors = collect_plan_validation_errors(plan)
    assert any(".." in error for error in errors)

    payload = _valid_plan_dict()
    payload["file_manifest"][1]["path"] = "/etc/passwd"
    plan = ProjectPlan.model_validate(payload)
    errors = collect_plan_validation_errors(plan)
    assert any("relative" in error for error in errors)


def test_plan_without_tests_is_allowed() -> None:
    """Automated tests and paired source/test tasks are optional."""
    payload = _valid_plan_dict()
    payload["tasks"] = [
        task for task in payload["tasks"] if task["task_type"] != "test"
    ]
    payload["file_manifest"] = [
        spec for spec in payload["file_manifest"] if spec["file_type"] != "test"
    ]
    # Keep a dependency config file from the original valid plan.
    plan = ProjectPlan.model_validate(payload)
    assert collect_plan_validation_errors(plan) == []


def test_plan_without_dependency_config_is_allowed() -> None:
    """Dependency-configuration files are optional."""
    payload = _valid_plan_dict()
    payload["file_manifest"] = [
        spec
        for spec in payload["file_manifest"]
        if spec["path"] not in {"pyproject.toml", "requirements.txt", "package.json"}
    ]
    plan = ProjectPlan.model_validate(payload)
    assert collect_plan_validation_errors(plan) == []


def test_static_markup_plan_allows_missing_tests_and_deps() -> None:
    """HTML/static-site plans may omit automated tests and package configs."""
    from codegen_workflow.schemas.plan import is_static_markup_project

    payload = {
        "project_name": "marketplace_landing",
        "objective": "Create a simple landing page to expose a marketplace.",
        "assumptions": ["Static hosting only."],
        "language": "HTML",
        "framework": None,
        "architecture_pattern": "static site",
        "dependencies": [],
        "epics": [
            {
                "id": "epic-1",
                "title": "Landing page",
                "description": "Build the marketplace landing page.",
                "acceptance_criteria": [
                    "index.html exists and includes a products section heading."
                ],
            }
        ],
        "stories": [
            {
                "id": "story-1",
                "epic_id": "epic-1",
                "title": "Page structure",
                "description": "As a visitor I want a clear landing page layout.",
                "acceptance_criteria": [
                    "index.html contains header, main, and footer landmarks."
                ],
            },
            {
                "id": "story-2",
                "epic_id": "epic-1",
                "title": "Page styling",
                "description": "As a visitor I want readable visual styling.",
                "acceptance_criteria": [
                    "styles.css is linked from index.html and sets body font-size."
                ],
            },
        ],
        "tasks": [
            {
                "id": "task-1",
                "story_id": "story-1",
                "title": "Write HTML",
                "description": "Create the landing page HTML structure.",
                "task_type": "source",
                "dependencies": [],
                "files": ["index.html"],
                "acceptance_criteria": [
                    "index.html file exists with marketplace product section."
                ],
            },
            {
                "id": "task-2",
                "story_id": "story-2",
                "title": "Write CSS",
                "description": "Create stylesheet for the landing page.",
                "task_type": "source",
                "dependencies": ["task-1"],
                "files": ["styles.css"],
                "acceptance_criteria": [
                    "styles.css exists and is referenced by index.html."
                ],
            },
            {
                "id": "task-3",
                "story_id": "story-1",
                "title": "Write README",
                "description": "Document how to open the static site.",
                "task_type": "documentation",
                "dependencies": ["task-1"],
                "files": ["README.md"],
                "acceptance_criteria": [
                    "README.md explains how to open index.html locally."
                ],
            },
        ],
        "file_manifest": [
            {
                "path": "index.html",
                "purpose": "Landing page markup.",
                "file_type": "source",
                "requirements": ["Contains marketplace sections."],
                "depends_on": [],
            },
            {
                "path": "styles.css",
                "purpose": "Landing page styles.",
                "file_type": "source",
                "requirements": ["Linked from index.html."],
                "depends_on": ["index.html"],
            },
            {
                "path": "README.md",
                "purpose": "Project documentation.",
                "file_type": "documentation",
                "requirements": ["Describe how to view the page."],
                "depends_on": [],
            },
        ],
        "install_commands": [],
        "validation_commands": [
            'python3 -c "from pathlib import Path; assert Path(\'index.html\').is_file()"'
        ],
        "run_command": None,
        "risks": [],
    }
    plan = ProjectPlan.model_validate(payload)
    assert is_static_markup_project(plan) is True
    assert collect_plan_validation_errors(plan) == []
    assert validate_plan(plan) is plan


def test_unsafe_project_name_rejected_by_schema() -> None:
    """Project names with path separators fail schema validation."""
    payload = _valid_plan_dict(project_name="../evil")
    with pytest.raises(ValidationError):
        ProjectPlan.model_validate(payload)


def test_empty_validation_commands_rejected() -> None:
    """Empty validation_commands fail both schema and domain checks."""
    payload = _valid_plan_dict(validation_commands=[])
    with pytest.raises(ValidationError):
        ProjectPlan.model_validate(payload)
