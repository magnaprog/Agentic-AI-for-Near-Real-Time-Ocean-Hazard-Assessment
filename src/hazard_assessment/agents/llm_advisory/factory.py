"""LLM model factory - one hosted provider for v0.1.

Constructs a LangChain chat model from LLMSettings. Additional providers
can be added by extending the if-branch and installing the corresponding
langchain-* package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from langchain_core.language_models import BaseChatModel

if TYPE_CHECKING:
    from hazard_assessment.config.settings import LLMSettings


def build_chat_model(settings: LLMSettings, *, purpose: str = "standard") -> BaseChatModel:
    """Create a LangChain chat model for the configured provider.

    Args:
        settings: LLM configuration (provider, model name, API key, etc.).
        purpose: Model tier - ``"fast"`` uses ``fast_model`` if set,
            ``"quality"`` uses ``quality_model`` if set, ``"standard"``
            uses the default ``model``.

    Returns:
        A configured ``BaseChatModel`` with retry behaviour attached.

    Raises:
        ValueError: If no model identifier is configured, or if the
            provider is not the supported one.
    """
    model_name = _resolve_model(settings, purpose)

    if not model_name:
        raise ValueError(
            "No LLM model identifier configured. Set LLM_MODEL (or the "
            f"purpose-specific override for {purpose!r}) before enabling the LLM layer."
        )

    if settings.provider != "anthropic":
        raise ValueError(
            f"Unsupported LLM provider: {settings.provider!r}. "
            f"Only 'anthropic' is supported in v0.1."
        )

    from langchain_anthropic import ChatAnthropic

    # ChatAnthropic constructor signature varies across langchain-anthropic
    # versions - use **kwargs to avoid version-specific type errors.
    kwargs: dict[str, Any] = {
        "model": model_name,
        "api_key": settings.api_key,
        "timeout": float(settings.timeout_sec),
        "max_retries": 0,  # retry handled below via .with_retry()
    }
    llm = ChatAnthropic(**kwargs)

    # Only retry transient errors (timeouts, rate limits, server errors).
    # Auth, validation, and bad-request errors should fail immediately.
    from anthropic import (
        APITimeoutError,
        InternalServerError,
        RateLimitError,
    )

    return cast(
        BaseChatModel,
        llm.with_retry(
            retry_if_exception_type=(APITimeoutError, RateLimitError, InternalServerError),
            stop_after_attempt=settings.max_retries,
            wait_exponential_jitter=True,
        ),
    )


def _resolve_model(settings: LLMSettings, purpose: str) -> str:
    """Pick the model name based on purpose and settings."""
    if purpose == "fast" and settings.fast_model:
        return settings.fast_model
    if purpose == "quality" and settings.quality_model:
        return settings.quality_model
    return settings.model
