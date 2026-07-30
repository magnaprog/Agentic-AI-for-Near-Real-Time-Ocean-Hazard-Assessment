"""Chat-model factory.

Constructs a LangChain chat model from LLMSettings for whichever provider is
configured. This module names no vendor; the providers module holds the
registry, and adding a provider is a data change there.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

from langchain_core.language_models import BaseChatModel

if TYPE_CHECKING:
    from hazard_assessment.config.settings import LLMSettings

#: Stand-in credential for an operator-run endpoint that needs none. Chosen to
#: read as deliberate in a stack trace or request log.
_PLACEHOLDER_API_KEY = "not-required-for-local-endpoint"


def build_chat_model(
    settings: LLMSettings,
    *,
    purpose: str = "standard",
    tools: Sequence[Any] | None = None,
) -> BaseChatModel:
    """Create a LangChain chat model for the configured provider.

    Args:
        settings: LLM configuration (provider, model name, API key, etc.).
        purpose: Model tier - ``"fast"`` uses ``fast_model`` if set,
            ``"quality"`` uses ``quality_model`` if set, ``"standard"``
            uses the default ``model``.
        tools: Tools to bind. Binding has to happen here rather than on the
            returned object, because the retry wrapper this function applies is
            a RunnableRetry, which has no ``bind_tools``. Calling it on the
            result raised AttributeError before any request was made.

    Returns:
        A configured ``BaseChatModel`` with retry behavior attached.

    Raises:
        ValueError: If no model identifier is configured, if the provider is
            not in the registry, or if a requested base URL was not applied.
        ImportError: If the provider's integration package is not installed.
    """
    from hazard_assessment.agents.llm_advisory.providers import (
        base_url_applied,
        load_chat_model_class,
        resolve_spec,
        retryable_exceptions,
    )

    model_name = _resolve_model(settings, purpose)

    if not model_name:
        raise ValueError(
            "No LLM model identifier configured. Set LLM_MODEL (or the "
            f"purpose-specific override for {purpose!r}) before enabling the LLM layer."
        )

    spec = resolve_spec(settings.provider)
    chat_model_class = load_chat_model_class(spec)

    # These names are accepted by every registered integration, in some cases
    # as an alias over a vendor-specific field. Passed as **kwargs because the
    # underlying field names differ between them.
    kwargs: dict[str, Any] = {
        "model": model_name,
        "timeout": float(settings.timeout_sec),
        "max_retries": 0,  # retry handled below via .with_retry()
    }
    if settings.base_url:
        kwargs["base_url"] = settings.base_url
    if settings.api_key:
        kwargs["api_key"] = settings.api_key
    elif settings.base_url:
        # Locally served models generally want no credential, but the vendor
        # SDKs refuse to construct without one: openai raises "Missing
        # credentials" before any request is made. Since the endpoint is the
        # operator's own, send a placeholder rather than make them invent one.
        # It is not a secret, and against a real endpoint it simply fails
        # authentication with the usual error.
        kwargs["api_key"] = _PLACEHOLDER_API_KEY

    llm = chat_model_class(**kwargs)

    # These models ignore arguments they do not recognize, so a base URL can be
    # dropped in silence. Refusing here is the difference between a clear
    # failure and quietly sending prompts to the vendor's public endpoint when
    # the operator asked for their own.
    if settings.base_url and not base_url_applied(llm, settings.base_url):
        raise ValueError(
            f"{spec.module}.{spec.class_name} did not apply LLM_BASE_URL="
            f"{settings.base_url!r}. Refusing to run against a different "
            "endpoint than the one configured."
        )

    # Bind before wrapping: with_retry returns a RunnableRetry, and tools must
    # be attached to the chat model underneath it.
    bound = llm.bind_tools(tools) if tools else llm

    # Only retry transient failures. Authentication, validation and
    # bad-request errors must fail immediately.
    return cast(
        BaseChatModel,
        bound.with_retry(
            retry_if_exception_type=retryable_exceptions(spec),
            # stop_after_attempt counts total attempts, while the setting is
            # named and documented as retries, so the first attempt has to be
            # added. Passing max_retries directly made LLM_MAX_RETRIES=2 mean
            # one retry, and LLM_MAX_RETRIES=1 mean none at all despite the
            # field's ge=1 floor.
            stop_after_attempt=settings.max_retries + 1,
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
