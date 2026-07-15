"""Shared Azure OpenAI chat model construction for agent nodes."""

from __future__ import annotations

import os
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel


def _resolve_azure_endpoint() -> str | None:
    """Resolve the Azure OpenAI resource endpoint from the environment.

    Prefers ``AZURE_OPENAI_ENDPOINT``. Falls back to ``OPENAI_BASE_URL`` and
    strips a trailing ``/openai/...`` path when present so LangChain receives
    the bare resource URL Azure expects.

    Returns:
        Azure endpoint URL, or ``None`` when unset.
    """
    azure = (os.environ.get("AZURE_OPENAI_ENDPOINT") or "").strip().rstrip("/")
    if azure:
        return azure

    base = (os.environ.get("OPENAI_BASE_URL") or "").strip().rstrip("/")
    if not base:
        return None
    marker = "/openai"
    if marker in base:
        return base.split(marker, 1)[0]
    return base


def create_azure_chat_model(
    *,
    model_name: str,
    temperature: float = 0.0,
) -> BaseChatModel:
    """Create an ``AzureChatOpenAI`` client from environment settings.

    Args:
        model_name: Azure deployment name.
        temperature: Sampling temperature.

    Returns:
        Configured Azure chat model instance.

    Raises:
        ValueError: If required Azure settings are missing.
    """
    from langchain_openai import AzureChatOpenAI

    api_key = (
        os.environ.get("AZURE_OPENAI_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or None
    )
    endpoint = _resolve_azure_endpoint()
    api_version = (
        os.environ.get("OPENAI_API_VERSION")
        or os.environ.get("AZURE_OPENAI_API_VERSION")
        or None
    )

    missing: list[str] = []
    if not api_key:
        missing.append("AZURE_OPENAI_API_KEY (or OPENAI_API_KEY)")
    if not endpoint:
        missing.append("AZURE_OPENAI_ENDPOINT (or OPENAI_BASE_URL)")
    if not api_version:
        missing.append("OPENAI_API_VERSION (or AZURE_OPENAI_API_VERSION)")
    if missing:
        raise ValueError(
            "Missing Azure OpenAI configuration: " + ", ".join(missing)
        )

    kwargs: dict[str, Any] = {
        "azure_deployment": model_name,
        "azure_endpoint": endpoint,
        "api_key": api_key,
        "api_version": api_version,
        "temperature": temperature,
    }
    return AzureChatOpenAI(**kwargs)


# Backward-compatible alias used by older imports.
create_chat_openai = create_azure_chat_model
