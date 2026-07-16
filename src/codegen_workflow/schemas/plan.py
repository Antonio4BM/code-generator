"""Pydantic schemas and deterministic validators for project plans."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

TaskType = Literal[
    "configuration",
    "source",
    "test",
    "documentation",
    "container",
    "validation",
]

FileType = Literal[
    "source",
    "test",
    "configuration",
    "documentation",
    "container",
    "other",
]

# Filesystem-safe project name: starts with a letter; alphanumerics, _ and -.
_PROJECT_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")

# Vague acceptance-criteria patterns that are not measurable.
_VAGUE_CRITERION_RE = re.compile(
    r"^\s*("
    r"works(\s+well)?|"
    r"looks\s+good|"
    r"feels\s+(good|right)|"
    r"is\s+(good|fine|ok|okay|done|complete)|"
    r"as\s+expected|"
    r"correctly|"
    r"properly|"
    r"should\s+work|"
    r"etc\.?"
    r")\s*\.?\s*$",
    re.IGNORECASE,
)

_DOC_PATH_HINTS = ("readme", "docs/", "documentation")
_DEP_CONFIG_NAMES = {
    "requirements.txt",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "pipfile",
    "poetry.lock",
    "package.json",
    "package-lock.json",
    "yarn.lock",
    "go.mod",
    "cargo.toml",
    "gemfile",
    "composer.json",
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
}

# Languages / architectures treated as markup-first (tests & package managers optional).
_STATIC_MARKUP_LANGUAGES = frozenset(
    {
        "html",
        "css",
        "markdown",
        "md",
        "static",
        "static html",
        "static-site",
        "static site",
    }
)
_STATIC_ARCH_HINTS = (
    "static",
    "static site",
    "static-site",
    "static html",
    "landing page",
    "markup",
)
_STATIC_SOURCE_SUFFIXES = (
    ".html",
    ".htm",
    ".css",
    ".scss",
    ".sass",
    ".less",
    ".md",
    ".txt",
    ".svg",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".ico",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
)
# Optional static JS helpers still count as markup-site assets when language is HTML.
_STATIC_OPTIONAL_SCRIPT_SUFFIXES = (".js", ".mjs", ".ts")



class PlanValidationError(ValueError):
    """Raised when a project plan fails deterministic validation."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        message = "; ".join(errors) if errors else "plan validation failed"
        super().__init__(message)


class Epic(BaseModel):
    """High-level delivery theme derived from the user requirement."""

    id: str = Field(..., min_length=1, description="Unique epic identifier.")
    title: str = Field(..., min_length=1, description="Short epic title.")
    description: str = Field(..., min_length=1, description="Epic scope summary.")
    acceptance_criteria: list[str] = Field(
        ...,
        min_length=1,
        description="Measurable epic-level acceptance criteria.",
    )


class UserStory(BaseModel):
    """User-facing or technical story belonging to an epic."""

    id: str = Field(..., min_length=1, description="Unique story identifier.")
    epic_id: str = Field(..., min_length=1, description="Parent epic id.")
    title: str = Field(..., min_length=1, description="Short story title.")
    description: str = Field(
        ...,
        min_length=1,
        description="Story description; user-story format when appropriate.",
    )
    acceptance_criteria: list[str] = Field(
        ...,
        min_length=1,
        description="Measurable story-level acceptance criteria.",
    )


class ImplementationTask(BaseModel):
    """Concrete implementation unit for the coder agent."""

    id: str = Field(..., min_length=1, description="Unique task identifier.")
    story_id: str = Field(..., min_length=1, description="Parent story id.")
    title: str = Field(..., min_length=1, description="Short task title.")
    description: str = Field(..., min_length=1, description="What to implement.")
    task_type: TaskType = Field(..., description="Category of implementation work.")
    dependencies: list[str] = Field(
        default_factory=list,
        description="Task ids that must complete before this task.",
    )
    files: list[str] = Field(
        default_factory=list,
        description="Relative paths this task creates or updates.",
    )
    acceptance_criteria: list[str] = Field(
        ...,
        min_length=1,
        description="Measurable task-level acceptance criteria.",
    )


class FileSpecification(BaseModel):
    """Manifest entry for a file the coder must produce."""

    path: str = Field(..., min_length=1, description="Path relative to project root.")
    purpose: str = Field(..., min_length=1, description="Why the file exists.")
    file_type: FileType = Field(..., description="Logical file category.")
    requirements: list[str] = Field(
        default_factory=list,
        description="Content or behavioral requirements for the file.",
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description="Other relative paths this file depends on.",
    )


class ProjectPlan(BaseModel):
    """Complete validated plan for a greenfield software project."""

    project_name: str = Field(..., min_length=1, description="Filesystem-safe name.")
    objective: str = Field(..., min_length=1, description="Primary project goal.")
    assumptions: list[str] = Field(
        default_factory=list,
        description="Assumptions made while planning.",
    )
    language: str = Field(
        ..., min_length=1, description="Primary programming language."
    )
    framework: str | None = Field(
        default=None,
        description="Optional framework; null when none is justified.",
    )
    architecture_pattern: str = Field(
        ...,
        min_length=1,
        description="Chosen architecture style (for example modular monolith).",
    )
    dependencies: list[str] = Field(
        default_factory=list,
        description="Third-party packages or libraries to install.",
    )
    epics: list[Epic] = Field(..., min_length=1)
    stories: list[UserStory] = Field(..., min_length=1)
    tasks: list[ImplementationTask] = Field(..., min_length=1)
    file_manifest: list[FileSpecification] = Field(..., min_length=1)
    install_commands: list[str] = Field(
        default_factory=list,
        description="Commands to install dependencies.",
    )
    validation_commands: list[str] = Field(
        ...,
        min_length=1,
        description="Commands that verify the generated project.",
    )
    run_command: str | None = Field(
        default=None,
        description="Primary command to run the application, if any.",
    )
    risks: list[str] = Field(
        default_factory=list,
        description="Known risks or open questions.",
    )

    @field_validator("project_name")
    @classmethod
    def project_name_must_be_filesystem_safe(cls, value: str) -> str:
        """Reject project names that are not filesystem-safe."""
        if not _PROJECT_NAME_RE.fullmatch(value):
            raise ValueError(
                "project_name must start with a letter and contain only "
                "letters, digits, underscores, and hyphens"
            )
        return value


def _is_measurable_criterion(criterion: str) -> bool:
    """Return whether an acceptance criterion is non-empty and measurable."""
    text = criterion.strip()
    if len(text) < 8:
        return False
    if _VAGUE_CRITERION_RE.fullmatch(text):
        return False
    return True


def _has_cycle(task_ids: set[str], dependencies: dict[str, list[str]]) -> bool:
    """Detect a cycle among task dependency edges using DFS coloring."""
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {task_id: WHITE for task_id in task_ids}

    def visit(node: str) -> bool:
        color[node] = GRAY
        for neighbor in dependencies.get(node, []):
            if neighbor not in color:
                continue
            if color[neighbor] == GRAY:
                return True
            if color[neighbor] == WHITE and visit(neighbor):
                return True
        color[node] = BLACK
        return False

    return any(color[node] == WHITE and visit(node) for node in task_ids)


def _is_documentation_file(spec: FileSpecification) -> bool:
    """Return whether a manifest entry counts as project documentation."""
    path_lower = spec.path.lower().replace("\\", "/")
    if spec.file_type == "documentation":
        return True
    return any(hint in path_lower for hint in _DOC_PATH_HINTS)


def _is_dependency_config(spec: FileSpecification) -> bool:
    """Return whether a manifest entry is a dependency configuration file."""
    path_lower = spec.path.lower().replace("\\", "/")
    basename = path_lower.rsplit("/", 1)[-1]
    if basename in _DEP_CONFIG_NAMES:
        return True
    return spec.file_type == "configuration" and basename in {
        "requirements.txt",
        "pyproject.toml",
        "package.json",
        "go.mod",
        "cargo.toml",
    }


def _is_test_file(spec: FileSpecification) -> bool:
    """Return whether a manifest entry is an automated test file."""
    if spec.file_type == "test":
        return True
    path_lower = spec.path.lower().replace("\\", "/")
    basename = path_lower.rsplit("/", 1)[-1]
    return (
        path_lower.startswith("tests/")
        or "/tests/" in path_lower
        or basename.startswith("test_")
        or basename.endswith("_test.py")
        or basename.endswith(".test.js")
        or basename.endswith(".spec.ts")
        or basename.endswith("_test.go")
    )


def _is_static_asset_path(path: str) -> bool:
    """Return whether a path looks like a static-site asset."""
    lowered = path.lower().replace("\\", "/")
    basename = lowered.rsplit("/", 1)[-1]
    if not basename or basename.endswith("/"):
        return True
    suffixes = _STATIC_SOURCE_SUFFIXES + _STATIC_OPTIONAL_SCRIPT_SUFFIXES
    return any(basename.endswith(suffix) for suffix in suffixes) or basename in {
        "robots.txt",
        "favicon.ico",
        "sitemap.xml",
        "manifest.json",
    }


def is_static_markup_project(plan: ProjectPlan) -> bool:
    """Return whether the plan describes a static markup / content site.

    Used for prompting and heuristics. Automated tests and dependency
    configuration are optional for all plans; this helper still identifies
    markup-first projects.

    Args:
        plan: Candidate project plan.

    Returns:
        ``True`` when language/architecture and file manifest indicate a
        static markup site rather than an application codebase.
    """
    language = (plan.language or "").strip().lower()
    architecture = (plan.architecture_pattern or "").strip().lower()
    language_static = language in _STATIC_MARKUP_LANGUAGES
    architecture_static = any(hint in architecture for hint in _STATIC_ARCH_HINTS)
    if not (language_static or architecture_static):
        return False

    if plan.dependencies:
        return False
    if any(_is_dependency_config(spec) for spec in plan.file_manifest):
        return False

    sourceish = [
        spec
        for spec in plan.file_manifest
        if spec.file_type in {"source", "other", "configuration"}
        and not _is_documentation_file(spec)
        and not _is_test_file(spec)
    ]
    if not sourceish:
        return language_static or architecture_static

    return all(_is_static_asset_path(spec.path) for spec in sourceish)


def collect_plan_validation_errors(plan: ProjectPlan) -> list[str]:
    """Return deterministic validation errors for a project plan.

    Automated tests, paired source/test tasks, and dependency-configuration
    files are optional so static→API revisions can proceed with simpler
    plans. Documentation and non-empty validation_commands remain required.

    Args:
        plan: Structured plan produced by the planner model.

    Returns:
        A list of human-readable validation messages. Empty means valid.
    """
    errors: list[str] = []

    if not _PROJECT_NAME_RE.fullmatch(plan.project_name):
        errors.append(f"project_name is not filesystem-safe: {plan.project_name!r}")

    epic_ids = [epic.id for epic in plan.epics]
    story_ids = [story.id for story in plan.stories]
    task_ids = [task.id for task in plan.tasks]

    if len(epic_ids) != len(set(epic_ids)):
        errors.append("epic ids must be unique")
    if len(story_ids) != len(set(story_ids)):
        errors.append("story ids must be unique")
    if len(task_ids) != len(set(task_ids)):
        errors.append("task ids must be unique")

    epic_id_set = set(epic_ids)
    story_id_set = set(story_ids)
    task_id_set = set(task_ids)

    for story in plan.stories:
        if story.epic_id not in epic_id_set:
            errors.append(
                f"story {story.id!r} references missing epic {story.epic_id!r}"
            )

    dependency_map: dict[str, list[str]] = {}
    for task in plan.tasks:
        if task.story_id not in story_id_set:
            errors.append(
                f"task {task.id!r} references missing story {task.story_id!r}"
            )
        dependency_map[task.id] = list(task.dependencies)
        for dep in task.dependencies:
            if dep not in task_id_set:
                errors.append(f"task {task.id!r} depends on missing task {dep!r}")

    if task_id_set and _has_cycle(task_id_set, dependency_map):
        errors.append("task dependencies contain a cycle")

    seen_paths: set[str] = set()
    for spec in plan.file_manifest:
        path = spec.path.replace("\\", "/")
        if path.startswith("/") or re.match(r"^[A-Za-z]:/", path):
            errors.append(f"file path must be relative: {spec.path!r}")
        if ".." in path.split("/"):
            errors.append(f"file path must not contain '..': {spec.path!r}")
        if path in seen_paths:
            errors.append(f"duplicate file path: {spec.path!r}")
        seen_paths.add(path)

    def check_criteria(label: str, criteria: list[str]) -> None:
        for index, criterion in enumerate(criteria):
            if not _is_measurable_criterion(criterion):
                errors.append(
                    f"{label} acceptance criterion[{index}] is not measurable: "
                    f"{criterion!r}"
                )

    for epic in plan.epics:
        check_criteria(f"epic {epic.id}", epic.acceptance_criteria)
    for story in plan.stories:
        check_criteria(f"story {story.id}", story.acceptance_criteria)
    for task in plan.tasks:
        check_criteria(f"task {task.id}", task.acceptance_criteria)

    if not plan.validation_commands:
        errors.append("validation_commands must not be empty")

    if not any(_is_documentation_file(spec) for spec in plan.file_manifest):
        errors.append("file_manifest must include project documentation")

    return errors


def validate_plan(plan: ProjectPlan) -> ProjectPlan:
    """Validate a project plan and return it when valid.

    Args:
        plan: Structured plan to validate.

    Returns:
        The same plan instance when validation succeeds.

    Raises:
        PlanValidationError: When one or more validation rules fail.
    """
    errors = collect_plan_validation_errors(plan)
    if errors:
        raise PlanValidationError(errors)
    return plan
