"""Structured logging helpers for the API layer."""

from __future__ import annotations

import logging
import re
from typing import Any
from uuid import uuid4

_SECRET_PATTERNS = (
    re.compile(r"(?i)(api[_-]?key|authorization|bearer|token|password|secret)\s*[:=]\s*\S+"),
    re.compile(r"(?i)sk-[A-Za-z0-9]{10,}"),
)


class ContextFormatter(logging.Formatter):
    """Formatter that supplies default correlation fields when absent."""

    def format(self, record: logging.LogRecord) -> str:
        """Format a record, filling missing API correlation attributes.

        Args:
            record: Log record being emitted.

        Returns:
            Formatted log line.
        """
        if not hasattr(record, "request_id"):
            record.request_id = "-"
        if not hasattr(record, "workflow_id"):
            record.workflow_id = "-"
        if not hasattr(record, "endpoint"):
            record.endpoint = "-"
        return super().format(record)


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging for structured API messages.

    Correlation fields are filled by :class:`ContextFormatter` so non-API
    loggers (planner/coder/reviewer) do not crash the logging subsystem.

    Args:
        level: Logging level name such as ``INFO`` or ``DEBUG``.
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    formatter = ContextFormatter(
        "%(asctime)s %(levelname)s %(name)s "
        "request_id=%(request_id)s workflow_id=%(workflow_id)s "
        "endpoint=%(endpoint)s %(message)s"
    )
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        handler.addFilter(RequestContextFilter())
        root.addHandler(handler)
    else:
        for handler in root.handlers:
            handler.setFormatter(formatter)
            handler.addFilter(RequestContextFilter())
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def new_request_id() -> str:
    """Generate a short correlation identifier for a request.

    Returns:
        A UUID4 string.
    """
    return str(uuid4())


def redact_secrets(text: str) -> str:
    """Remove likely secrets from a log string.

    Args:
        text: Raw text that may contain credentials.

    Returns:
        Redacted text safe for logs.
    """
    redacted = text
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


class RequestContextFilter(logging.Filter):
    """Inject default correlation fields onto log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Ensure correlation attributes always exist.

        Args:
            record: Log record being emitted.

        Returns:
            Always ``True`` so the record is emitted.
        """
        if not hasattr(record, "request_id"):
            record.request_id = "-"
        if not hasattr(record, "workflow_id"):
            record.workflow_id = "-"
        if not hasattr(record, "endpoint"):
            record.endpoint = "-"
        return True


def log_extra(
    *,
    request_id: str | None = None,
    workflow_id: str | None = None,
    endpoint: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build a logging ``extra`` dict with correlation fields.

    Args:
        request_id: Per-request correlation identifier.
        workflow_id: Workflow / thread identifier when known.
        endpoint: HTTP path or logical endpoint name.
        **kwargs: Additional structured fields.

    Returns:
        Dictionary suitable for ``logger.info(..., extra=...)``.
    """
    payload: dict[str, Any] = {
        "request_id": request_id or "-",
        "workflow_id": workflow_id or "-",
        "endpoint": endpoint or "-",
    }
    payload.update(kwargs)
    return payload
