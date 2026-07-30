"""Tests for the chat-model provider registry and factory.

This surface had no coverage, which is how a real defect survived: the factory
returned the result of ``with_retry``, a RunnableRetry with no ``bind_tools``,
and the after-action graph then called ``bind_tools`` on it. Building that
graph raised AttributeError before reaching a model, so the only
model-directed loop in the system could not run.

The provider packages are optional and are not installed for the test run, so
the factory tests substitute a stand-in chat-model class. That keeps them
honest about what they check: the wiring in this repository, not a vendor SDK.
"""

from __future__ import annotations

from typing import Any

import pytest

from hazard_assessment.agents.llm_advisory import providers
from hazard_assessment.agents.llm_advisory.factory import build_chat_model
from hazard_assessment.config.settings import LLMSettings


class _FakeChatModel:
    """Stand-in with the parts the factory touches.

    ``model_fields`` mirrors pydantic so ``base_url_applied`` can find a value
    by comparison, and both binder methods return ``self`` so a test can see
    the order they were called in.
    """

    model_fields: dict[str, Any] = {
        "model": None,
        "api_key": None,
        "base_url": None,
        "timeout": None,
        "max_retries": None,
    }

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = dict(kwargs)
        self.calls: list[str] = []
        self.bound_tools: Any = None
        self.retry_kwargs: dict[str, Any] = {}
        for key, value in kwargs.items():
            setattr(self, key, value)

    def bind_tools(self, tools: Any) -> _FakeChatModel:
        self.calls.append("bind_tools")
        self.bound_tools = tools
        return self

    def with_retry(self, **kwargs: Any) -> _FakeChatModel:
        self.calls.append("with_retry")
        self.retry_kwargs = kwargs
        return self


@pytest.fixture
def fake_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every registered provider construct the stand-in class."""
    monkeypatch.setattr(
        providers, "load_chat_model_class", lambda spec: _FakeChatModel
    )


class TestRegistry:
    def test_supported_providers_are_sorted_and_known(self) -> None:
        assert providers.supported_providers() == (
            "anthropic",
            "google_genai",
            "openai",
        )

    def test_unknown_provider_names_the_alternatives(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            providers.resolve_spec("bedrock")
        message = str(excinfo.value)
        for name in providers.supported_providers():
            assert name in message
        # The escape hatch has to be discoverable from the error itself.
        assert "LLM_BASE_URL" in message

    def test_every_spec_resolves_to_a_real_class_path(self) -> None:
        for name in providers.supported_providers():
            spec = providers.resolve_spec(name)
            assert spec.module and spec.class_name and spec.distribution
            assert spec.retryable, f"{name} declares no retryable exceptions"

    def test_missing_package_error_names_the_distribution(self) -> None:
        spec = providers.ProviderSpec(
            distribution="not-a-real-distribution",
            module="hazard_assessment_absent_module",
            class_name="Chat",
            retryable=(),
        )
        with pytest.raises(ImportError, match="not-a-real-distribution"):
            providers.load_chat_model_class(spec)


class TestRetryableExceptions:
    def test_transport_failures_always_included(self) -> None:
        import httpx

        for name in providers.supported_providers():
            resolved = providers.retryable_exceptions(providers.resolve_spec(name))
            assert httpx.TimeoutException in resolved
            assert httpx.ConnectError in resolved

    def test_never_empty(self) -> None:
        """An empty tuple would silently disable retry while looking set up."""
        spec = providers.ProviderSpec(
            distribution="d", module="m", class_name="C",
            retryable=("nonexistent.module.SomeError",),
        )
        assert providers.retryable_exceptions(spec)

    def test_unresolvable_paths_are_skipped_not_raised(self) -> None:
        spec = providers.ProviderSpec(
            distribution="d", module="m", class_name="C",
            retryable=("httpx.NoSuchError", "httpx.TimeoutException"),
        )
        import httpx

        resolved = providers.retryable_exceptions(spec)
        assert httpx.TimeoutException in resolved

    def test_non_exception_paths_are_rejected(self) -> None:
        spec = providers.ProviderSpec(
            distribution="d", module="m", class_name="C",
            retryable=("httpx.Client",),  # a real class, but not an exception
        )
        import httpx

        assert httpx.Client not in providers.retryable_exceptions(spec)


class TestBaseUrlApplied:
    def test_true_when_a_field_holds_the_value(self) -> None:
        model = _FakeChatModel(base_url="http://localhost:1234/v1")
        assert providers.base_url_applied(model, "http://localhost:1234/v1")

    def test_false_when_the_value_was_dropped(self) -> None:
        assert not providers.base_url_applied(
            _FakeChatModel(model="m"), "http://localhost:1234/v1"
        )


class TestIsEnabled:
    def test_off_by_default(self) -> None:
        assert LLMSettings().is_enabled is False

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"api_key": "k"},
            {"base_url": "http://localhost:11434/v1"},
            {"api_key": "k", "base_url": "http://localhost:11434/v1"},
        ],
    )
    def test_on_when_a_key_or_an_endpoint_is_given(self, kwargs: dict[str, str]) -> None:
        assert LLMSettings(**kwargs).is_enabled is True

    def test_a_model_alone_does_not_enable_it(self) -> None:
        """A model name is not a request to start calling one."""
        assert LLMSettings(model="some-model").is_enabled is False


class TestFactory:
    def test_tools_are_bound_before_the_retry_wrapper(self, fake_provider: None) -> None:
        """The defect this file exists for: order matters.

        Binding after with_retry hits a RunnableRetry, which has no bind_tools.
        """
        model = build_chat_model(
            LLMSettings(provider="openai", model="m", api_key="k"),
            tools=["tool-a"],
        )
        assert model.calls == ["bind_tools", "with_retry"]  # type: ignore[attr-defined]
        assert model.bound_tools == ["tool-a"]  # type: ignore[attr-defined]

    def test_no_binding_when_no_tools_requested(self, fake_provider: None) -> None:
        model = build_chat_model(LLMSettings(provider="openai", model="m", api_key="k"))
        assert model.calls == ["with_retry"]  # type: ignore[attr-defined]

    def test_retry_budget_counts_the_first_attempt(self, fake_provider: None) -> None:
        model = build_chat_model(
            LLMSettings(provider="openai", model="m", api_key="k", max_retries=2)
        )
        assert model.retry_kwargs["stop_after_attempt"] == 3  # type: ignore[attr-defined]

    def test_canonical_kwargs_are_passed(self, fake_provider: None) -> None:
        model = build_chat_model(
            LLMSettings(provider="openai", model="gpt-x", api_key="k", timeout_sec=17)
        )
        kwargs = model.kwargs  # type: ignore[attr-defined]
        assert kwargs["model"] == "gpt-x"
        assert kwargs["api_key"] == "k"
        assert kwargs["timeout"] == 17.0
        # Vendor retry is off because with_retry owns it.
        assert kwargs["max_retries"] == 0

    def test_placeholder_key_for_a_keyless_endpoint(self, fake_provider: None) -> None:
        model = build_chat_model(
            LLMSettings(
                provider="openai", model="llama3", base_url="http://localhost:11434/v1"
            )
        )
        kwargs = model.kwargs  # type: ignore[attr-defined]
        assert kwargs["base_url"] == "http://localhost:11434/v1"
        # Present, because the vendor SDKs refuse to construct without one, and
        # obviously not a real credential.
        assert kwargs["api_key"]
        assert "local" in kwargs["api_key"]

    def test_a_real_key_is_not_overwritten(self, fake_provider: None) -> None:
        model = build_chat_model(
            LLMSettings(
                provider="openai", model="m", api_key="sk-real",
                base_url="http://gateway.internal/v1",
            )
        )
        assert model.kwargs["api_key"] == "sk-real"  # type: ignore[attr-defined]

    def test_dropped_base_url_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, fake_provider: None
    ) -> None:
        """A silently ignored base URL would send prompts to the public API."""
        monkeypatch.setattr(providers, "base_url_applied", lambda model, url: False)
        with pytest.raises(ValueError, match="did not apply LLM_BASE_URL"):
            build_chat_model(
                LLMSettings(
                    provider="openai", model="m", base_url="http://localhost:11434/v1"
                )
            )

    def test_unknown_provider_is_rejected(self, fake_provider: None) -> None:
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            build_chat_model(LLMSettings(provider="bedrock", model="m", api_key="k"))

    def test_missing_model_is_rejected(self, fake_provider: None) -> None:
        """A key without a model must fail loudly, not disable the layer."""
        with pytest.raises(ValueError, match="No LLM model identifier"):
            build_chat_model(LLMSettings(provider="openai", api_key="k"))

    def test_purpose_selects_the_override_model(self, fake_provider: None) -> None:
        settings = LLMSettings(
            provider="openai", model="base", api_key="k",
            fast_model="cheap", quality_model="good",
        )
        assert build_chat_model(settings, purpose="fast").kwargs["model"] == "cheap"  # type: ignore[attr-defined]
        assert build_chat_model(settings, purpose="quality").kwargs["model"] == "good"  # type: ignore[attr-defined]
        assert build_chat_model(settings).kwargs["model"] == "base"  # type: ignore[attr-defined]
