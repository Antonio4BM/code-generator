"""Environment-backed application configuration for the API."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _split_origins(raw: str) -> list[str]:
    """Parse a comma-separated CORS origin list.

    Args:
        raw: Environment value, possibly empty.

    Returns:
        List of trimmed origins. Empty string yields an empty list.
    """
    if not raw.strip():
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


@dataclass(frozen=True)
class APISettings:
    """Runtime settings loaded from environment variables.

    Attributes:
        openai_api_key: Model provider API key (never returned to clients).
        planner_model: Model name override for the planner agent.
        coder_model: Model name override for the coder agent.
        reviewer_model: Model name override for the reviewer agent.
        workspace_base_dir: Parent directory for isolated workspaces.
        artifact_base_dir: Directory used for artifact path containment checks.
        log_level: Structured logging level name.
        allowed_origins: CORS allowed origins (empty disables CORS middleware).
        app_env: Deployment environment name (``development`` / ``production``).
        workflow_timeout_seconds: Soft bound for synchronous graph invocations.
    """

    openai_api_key: str | None = None
    planner_model: str = "gpt-4.1-mini"
    coder_model: str = "gpt-4.1-mini"
    reviewer_model: str = "gpt-4.1-mini"
    workspace_base_dir: Path = field(default_factory=lambda: Path("workspaces"))
    artifact_base_dir: Path = field(default_factory=lambda: Path("artifacts"))
    log_level: str = "INFO"
    allowed_origins: list[str] = field(default_factory=list)
    app_env: str = "development"
    workflow_timeout_seconds: float = 600.0

    @property
    def is_production(self) -> bool:
        """Return whether the process is running in production."""
        return self.app_env.strip().lower() == "production"

    @classmethod
    def from_env(cls) -> APISettings:
        """Load settings from process environment variables.

        Returns:
            Populated :class:`APISettings` instance.
        """
        return cls(
            openai_api_key=os.environ.get("OPENAI_API_KEY") or None,
            planner_model=os.environ.get("PLANNER_MODEL", "gpt-4.1-mini"),
            coder_model=os.environ.get("CODER_MODEL", "gpt-4.1-mini"),
            reviewer_model=os.environ.get("REVIEWER_MODEL", "gpt-4.1-mini"),
            workspace_base_dir=Path(
                os.environ.get("WORKSPACE_BASE_DIR", "workspaces")
            ).resolve(),
            artifact_base_dir=Path(
                os.environ.get("ARTIFACT_BASE_DIR", "artifacts")
            ).resolve(),
            log_level=os.environ.get("LOG_LEVEL", "INFO").upper(),
            allowed_origins=_split_origins(os.environ.get("ALLOWED_ORIGINS", "")),
            app_env=os.environ.get("APP_ENV", "development"),
            workflow_timeout_seconds=float(
                os.environ.get("WORKFLOW_TIMEOUT_SECONDS", "600")
            ),
        )
