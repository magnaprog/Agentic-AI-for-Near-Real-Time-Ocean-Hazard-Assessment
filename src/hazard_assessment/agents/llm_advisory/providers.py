"""Chat-model provider registry.

The LLM layer is optional and advisory, but it should not be tied to one
vendor. Each entry below names a LangChain integration package and the
exception types from that vendor's SDK that are worth retrying. Adding a
provider is a data change; nothing else in this package names a vendor.

Constructor arguments do not need to be mapped per provider. The three
integrations here all accept ``model``, ``api_key``, ``base_url``, ``timeout``
and ``max_retries`` under those names, in some cases as pydantic aliases over
a vendor-specific field. That was checked against langchain-anthropic 1.5.3,
langchain-openai 1.4.1 and langchain-google-genai 4.3.2 rather than assumed.

``openai`` is also the general escape hatch. Any endpoint that speaks the
OpenAI chat-completions protocol works by pointing ``LLM_BASE_URL`` at it, so
local servers and aggregators need no entry of their own.

One caution behind the design. These chat models are pydantic models
configured with ``extra="ignore"``, so an argument a provider does not
recognize is dropped without complaint. That matters most for ``base_url``:
silently ignoring it would send prompts to the vendor's public endpoint
instead of the operator's own. ``base_url_applied`` exists so the caller can
refuse to proceed in that case rather than misroute the traffic.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProviderSpec:
    """How to build and retry one provider's chat model."""

    distribution: str
    """Package to install, used in the error message when the import fails."""

    module: str
    """Import path of the LangChain integration."""

    class_name: str
    """Chat-model class inside that module."""

    retryable: tuple[str, ...]
    """Dotted paths to exception types that indicate a transient failure."""


#: Transport-level failures, used when a provider's own SDK is unavailable.
#: httpx is a core dependency and underlies these SDKs, so these resolve even
#: when the vendor package does not.
_TRANSPORT_RETRYABLE: tuple[str, ...] = (
    "httpx.TimeoutException",
    "httpx.ConnectError",
)

_PROVIDERS: dict[str, ProviderSpec] = {
    "anthropic": ProviderSpec(
        distribution="langchain-anthropic",
        module="langchain_anthropic",
        class_name="ChatAnthropic",
        retryable=(
            "anthropic.APITimeoutError",
            "anthropic.APIConnectionError",
            "anthropic.RateLimitError",
            "anthropic.InternalServerError",
        ),
    ),
    "openai": ProviderSpec(
        distribution="langchain-openai",
        module="langchain_openai",
        class_name="ChatOpenAI",
        retryable=(
            "openai.APITimeoutError",
            "openai.APIConnectionError",
            "openai.RateLimitError",
            "openai.InternalServerError",
        ),
    ),
    "google_genai": ProviderSpec(
        distribution="langchain-google-genai",
        module="langchain_google_genai",
        class_name="ChatGoogleGenerativeAI",
        # google.genai.errors has no rate-limit type: ServerError covers 5xx,
        # while a 429 arrives as ClientError alongside authentication and
        # request errors. Retrying ClientError would retry a bad key forever,
        # so rate-limit retry is deliberately not covered for this provider.
        retryable=("google.genai.errors.ServerError",),
    ),
}


def supported_providers() -> tuple[str, ...]:
    """Provider names accepted in ``LLM_PROVIDER``."""
    return tuple(sorted(_PROVIDERS))


def resolve_spec(provider: str) -> ProviderSpec:
    """Look up a provider, raising with the accepted names if it is unknown."""
    try:
        return _PROVIDERS[provider]
    except KeyError:
        raise ValueError(
            f"Unsupported LLM provider: {provider!r}. Set LLM_PROVIDER to one of "
            f"{', '.join(supported_providers())}. Any OpenAI-compatible endpoint, "
            "including a locally served model, is reached with "
            "LLM_PROVIDER=openai and LLM_BASE_URL."
        ) from None


def load_chat_model_class(spec: ProviderSpec) -> type:
    """Import a provider's chat-model class, naming the package if absent."""
    try:
        module = importlib.import_module(spec.module)
    except ImportError as exc:
        raise ImportError(
            f"{spec.module} is required for LLM_PROVIDER support but is not "
            f"installed. Install it with: pip install {spec.distribution}"
        ) from exc
    return getattr(module, spec.class_name)  # type: ignore[no-any-return]


def retryable_exceptions(spec: ProviderSpec) -> tuple[type[BaseException], ...]:
    """Resolve a provider's transient-failure types.

    Paths that do not import are skipped, because a vendor renaming an
    exception should narrow retry coverage rather than break the LLM layer.
    Transport failures are always included, which also guarantees the result is
    never empty: an empty tuple would disable retry while looking configured.
    """
    resolved: list[type[BaseException]] = []
    for path in (*spec.retryable, *_TRANSPORT_RETRYABLE):
        module_path, _, class_name = path.rpartition(".")
        try:
            candidate = getattr(importlib.import_module(module_path), class_name)
        except (ImportError, AttributeError):
            logger.debug("Retryable exception %s unavailable; skipping", path)
            continue
        if isinstance(candidate, type) and issubclass(candidate, BaseException):
            resolved.append(candidate)
        else:
            logger.debug("Retryable path %s is not an exception type", path)
    return tuple(resolved)


def base_url_applied(model: object, base_url: str) -> bool:
    """Whether a requested base URL reached the constructed model.

    Each integration stores it under a different field name, so this looks for
    the value rather than a name. See the module docstring for why a silent
    drop must not be allowed to pass.
    """
    fields = getattr(type(model), "model_fields", None)
    if not fields:
        return False
    return any(getattr(model, name, None) == base_url for name in fields)
