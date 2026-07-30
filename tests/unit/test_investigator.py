"""Tests for the active-event evidence investigator and the shared tool loop.

The tool loop is the agentic mechanism and the place where the bounds live, so
it is tested directly rather than only through the investigator. It had no
coverage before: nothing exercised it, which is the same blind spot that let
the factory return an object with no ``bind_tools`` on it.

No provider package is installed for the test run, so the chat model is a
scripted stand-in. That keeps these tests about this repository's wiring, the
round cap, the call log and the guardrail decision, rather than a vendor SDK.
"""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

import pytest

from hazard_assessment.agents.llm_advisory import investigator as inv
from hazard_assessment.agents.llm_advisory.prompts import INVESTIGATOR_ISSUE_PROMPTS
from hazard_assessment.agents.llm_advisory.tools import resolve_tool_calls
from hazard_assessment.audit.logger import AuditLogger
from hazard_assessment.config.settings import LLMSettings

EVENT_ID = uuid4()


class _Reply:
    """Stands in for a chat-model response."""

    def __init__(self, content: str = "", tool_calls: list[dict[str, Any]] | None = None):
        self.content = content
        self.tool_calls = tool_calls or []


class _ScriptedModel:
    """Returns queued replies, then plain text once the script runs out."""

    def __init__(self, replies: list[_Reply]) -> None:
        self._replies = list(replies)
        self.invocations = 0

    def invoke(self, messages: Any) -> _Reply:
        self.invocations += 1
        if self._replies:
            return self._replies.pop(0)
        return _Reply("final answer")

    def bind_tools(self, tools: Any) -> _ScriptedModel:
        return self


class _RecordingTool:
    def __init__(self, name: str, *, raises: bool = False) -> None:
        self.name = name
        self._raises = raises
        self.calls: list[dict[str, Any]] = []

    def invoke(self, args: dict[str, Any]) -> str:
        self.calls.append(args)
        if self._raises:
            raise RuntimeError("tool exploded")
        return '{"entries": []}'


def _tool_call(name: str, call_id: str = "c1") -> dict[str, Any]:
    return {"name": name, "args": {}, "id": call_id}


class TestIssueSet:
    def test_issue_names_and_prompts_agree(self) -> None:
        """A named issue with no prompt would raise KeyError at run time."""
        assert set(inv.ISSUE_NAMES) == set(INVESTIGATOR_ISSUE_PROMPTS)

    def test_every_prompt_forbids_the_reserved_terms(self) -> None:
        """A finding is dropped whole on a violation, so each prompt warns."""
        for name, prompt in INVESTIGATOR_ISSUE_PROMPTS.items():
            assert "reserved for official NOAA products" in prompt, name

    def test_every_prompt_refuses_a_decision_role(self) -> None:
        for name, prompt in INVESTIGATOR_ISSUE_PROMPTS.items():
            assert "not deciding anything" in prompt, name


class TestDependencyProbe:
    """The endpoint's 501 has to depend on something that needs langchain.

    investigator.py imports langchain only inside function bodies, so importing
    it succeeds whether or not the provider stack is installed. The endpoint's
    guarded import therefore has to name a module that does import langchain at
    module level, or a missing dependency surfaces as three separately failed
    issues and an HTTP 200 instead of a 501.
    """

    def test_investigator_defers_its_langchain_imports(self) -> None:
        import ast
        import pathlib

        tree = ast.parse(
            pathlib.Path(inv.__file__).read_text(encoding="utf-8")
        )
        top_level = [
            n for n in tree.body if isinstance(n, ast.Import | ast.ImportFrom)
        ]
        names = [
            (n.module or "") if isinstance(n, ast.ImportFrom)
            else ",".join(a.name for a in n.names)
            for n in top_level
        ]
        assert not [n for n in names if n.startswith(("langchain", "langgraph"))]

    def test_the_factory_does_not(self) -> None:
        """So importing the factory is a genuine probe for the provider stack."""
        import ast
        import pathlib

        from hazard_assessment.agents.llm_advisory import factory

        tree = ast.parse(
            pathlib.Path(factory.__file__).read_text(encoding="utf-8")
        )
        module_level = [
            (n.module or "")
            for n in tree.body
            if isinstance(n, ast.ImportFrom)
        ]
        assert any(m.startswith("langchain") for m in module_level)


class TestInvocationIdentity:
    def test_shape_satisfies_the_column_check(self) -> None:
        """Migration 009 constrains this to 64 lowercase hex characters."""
        import re

        value = inv.compute_invocation_id(
            assessment_row_id=1, issue_name="station_agreement", model="m"
        )
        assert re.fullmatch(r"[0-9a-f]{64}", value)

    def test_deterministic(self) -> None:
        args = {"assessment_row_id": 3, "issue_name": "evidence_gaps", "model": "m"}
        assert inv.compute_invocation_id(**args) == inv.compute_invocation_id(**args)

    @pytest.mark.parametrize(
        "changed",
        [
            {"assessment_row_id": 4},
            {"issue_name": "timeline_consistency"},
            {"model": "other-model"},
            {"prompt_version": 99},
        ],
    )
    def test_every_component_changes_the_identity(self, changed: dict[str, Any]) -> None:
        """Anything that makes it a different finding must not collide."""
        base = {
            "assessment_row_id": 3,
            "issue_name": "evidence_gaps",
            "model": "m",
            "prompt_version": 1,
        }
        assert inv.compute_invocation_id(**base) != inv.compute_invocation_id(
            **{**base, **changed}
        )


class TestResultDigest:
    """The digest must cover the payload it is stored beside.

    It first covered only the finding text and the tool log, so
    guardrail_violations sat outside it: the one field recording that a finding
    was withheld was the one field a hash could not detect a change to.
    """

    def _finding(self, **overrides: Any) -> inv.IssueFinding:
        base: dict[str, Any] = {
            "issue_name": "station_agreement",
            "invocation_id": "a" * 64,
            "finding": "two of six stations agree",
            "tool_calls": [{"tool": "query_audit_trail"}],
            "guardrail_violations": [],
        }
        return inv.IssueFinding(**{**base, **overrides})

    def test_digest_is_the_hash_of_the_stored_payload(self) -> None:
        import hashlib
        import json

        finding = self._finding()
        expected = hashlib.sha256(
            json.dumps(
                finding.to_result_payload(),
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        assert finding.result_sha256 == expected

    @pytest.mark.parametrize(
        "overrides",
        [
            {"finding": "a different finding"},
            {"tool_calls": [{"tool": "query_fsm_transitions"}]},
            {"issue_name": "evidence_gaps"},
            {"guardrail_violations": ["Warning"]},
        ],
    )
    def test_every_stored_field_changes_the_digest(self, overrides: dict[str, Any]) -> None:
        assert self._finding().result_sha256 != self._finding(**overrides).result_sha256

    def test_shape_satisfies_the_column_check(self) -> None:
        import re

        assert re.fullmatch(r"[0-9a-f]{64}", self._finding().result_sha256)


class TestToolLoop:
    def test_returns_text_without_calling_tools(self) -> None:
        model = _ScriptedModel([])
        out = resolve_tool_calls(_Reply("straight answer"), model, [], call_log=[])
        assert out == "straight answer"
        assert model.invocations == 0

    def test_executes_a_requested_tool_and_stamps_the_node(self) -> None:
        tool = _RecordingTool("query_audit_trail")
        log: list[dict[str, Any]] = []
        model = _ScriptedModel([_Reply("answered")])
        out = resolve_tool_calls(
            _Reply("", [_tool_call("query_audit_trail")]),
            model, [tool], call_log=log, node="station_agreement",
        )
        assert out == "answered"
        assert tool.calls == [{}]

    def test_unknown_tool_is_logged_not_raised(self) -> None:
        log: list[dict[str, Any]] = []
        model = _ScriptedModel([_Reply("recovered")])
        out = resolve_tool_calls(
            _Reply("", [_tool_call("query_the_internet")]),
            model, [], call_log=log, node="evidence_gaps",
        )
        assert out == "recovered"
        assert log == [{
            "node": "evidence_gaps",
            "tool": "query_the_internet",
            "args": {},
            "error": "unknown_tool",
        }]

    def test_tool_failure_is_logged_and_reported_to_the_model(self) -> None:
        tool = _RecordingTool("query_audit_trail", raises=True)
        log: list[dict[str, Any]] = []
        model = _ScriptedModel([_Reply("noted the failure")])
        out = resolve_tool_calls(
            _Reply("", [_tool_call("query_audit_trail")]),
            model, [tool], call_log=log, node="n",
        )
        assert out == "noted the failure"
        assert log[0]["error"] == "RuntimeError"

    def test_non_convergence_is_reported_not_returned_as_an_answer(self) -> None:
        """A caller cannot tell a truncated analysis from a complete one."""
        tool = _RecordingTool("query_audit_trail")
        log: list[dict[str, Any]] = []
        # Always asks for another tool call, so the cap is what stops it.
        model = _ScriptedModel([_Reply("", [_tool_call("query_audit_trail")])] * 10)
        out = resolve_tool_calls(
            _Reply("", [_tool_call("query_audit_trail")]),
            model, [tool], max_rounds=2, call_log=log, node="n",
        )
        assert "did not converge" in out
        assert log[-1]["error"] == "tool_loop_did_not_converge"
        assert log[-1]["max_rounds"] == 2

    def test_the_round_cap_bounds_model_invocations(self) -> None:
        tool = _RecordingTool("query_audit_trail")
        model = _ScriptedModel([_Reply("", [_tool_call("query_audit_trail")])] * 10)
        resolve_tool_calls(
            _Reply("", [_tool_call("query_audit_trail")]),
            model, [tool], max_rounds=3, call_log=[],
        )
        assert model.invocations == 3


class TestInvestigateIssue:
    @pytest.fixture
    def settings(self) -> LLMSettings:
        return LLMSettings(provider="openai", model="m", api_key="k")

    def _patch_model(self, monkeypatch: pytest.MonkeyPatch, replies: list[_Reply]) -> None:
        from hazard_assessment.agents.llm_advisory import factory

        monkeypatch.setattr(
            factory, "build_chat_model", lambda *a, **k: _ScriptedModel(replies)
        )

    def test_records_a_finding(
        self, monkeypatch: pytest.MonkeyPatch, settings: LLMSettings
    ) -> None:
        self._patch_model(monkeypatch, [_Reply("two of six stations agree")])
        finding = inv.investigate_issue(
            settings, AuditLogger(), event_id=EVENT_ID,
            assessment_row_id=11, issue_name="station_agreement",
        )
        assert finding.finding == "two of six stations agree"
        assert finding.issue_name == "station_agreement"
        assert finding.guardrail_violations == []
        assert finding.to_result_payload()["prompt_version"] == inv.PROMPT_VERSION

    def test_reserved_terminology_drops_the_whole_finding(
        self, monkeypatch: pytest.MonkeyPatch, settings: LLMSettings
    ) -> None:
        """Partial redaction would misrepresent what the model said."""
        self._patch_model(monkeypatch, [_Reply("Issue a Tsunami Warning immediately.")])
        finding = inv.investigate_issue(
            settings, AuditLogger(), event_id=EVENT_ID,
            assessment_row_id=11, issue_name="station_agreement",
        )
        assert "finding withheld" in finding.finding
        assert "Warning" in finding.guardrail_violations
        assert "Tsunami Warning" not in finding.finding

    def test_result_digest_is_hex64_and_tracks_the_finding(
        self, monkeypatch: pytest.MonkeyPatch, settings: LLMSettings
    ) -> None:
        import re

        self._patch_model(monkeypatch, [_Reply("finding A")])
        first = inv.investigate_issue(
            settings, AuditLogger(), event_id=EVENT_ID,
            assessment_row_id=11, issue_name="evidence_gaps",
        )
        self._patch_model(monkeypatch, [_Reply("finding B")])
        second = inv.investigate_issue(
            settings, AuditLogger(), event_id=EVENT_ID,
            assessment_row_id=11, issue_name="evidence_gaps",
        )
        assert re.fullmatch(r"[0-9a-f]{64}", first.result_sha256)
        assert first.result_sha256 != second.result_sha256
        # Same assessment, issue and model, so the identity is unchanged: the
        # unique index is what refuses the second opinion.
        assert first.invocation_id == second.invocation_id

    def test_reserved_terms_in_the_tool_log_are_withheld_too(
        self, monkeypatch: pytest.MonkeyPatch, settings: LLMSettings
    ) -> None:
        """The model also authors tool names and arguments, and both are stored.

        Scanning only the finding let a reserved term ride through in the
        tool-call log while the finding itself came back clean.
        """
        reserved = "Tsunami Warning for the outer coast"
        self._patch_model(
            monkeypatch,
            [
                _Reply("", [{"name": reserved, "args": {}, "id": "c1"}]),
                _Reply("three of five stations produced a scored window"),
            ],
        )
        finding = inv.investigate_issue(
            settings, AuditLogger(), event_id=EVENT_ID,
            assessment_row_id=11, issue_name="station_agreement",
        )

        # The finding itself was clean and is kept.
        assert "three of five" in finding.finding
        # The log is withheld rather than rewritten, and the violation surfaces.
        assert "Warning" in finding.guardrail_violations
        payload = json.dumps(finding.to_result_payload())
        assert "Tsunami Warning" not in payload
        assert finding.tool_calls[0]["withheld"]
        assert finding.tool_calls[0]["n_calls"] == 1

    @pytest.mark.parametrize(("max_rounds", "expected_invocations"), [(1, 2), (2, 3), (3, 4)])
    def test_max_rounds_reaches_the_loop(
        self,
        monkeypatch: pytest.MonkeyPatch,
        settings: LLMSettings,
        max_rounds: int,
        expected_invocations: int,
    ) -> None:
        """investigate_assessment forwards it; it used to be unreachable.

        The budget is one invocation by the caller plus ``max_rounds`` inside
        the loop, so N rounds allow N+1 model calls. Pinned here because the
        parameter is the only bound on how much work one issue can ask for.
        """
        from hazard_assessment.agents.llm_advisory import factory

        models: list[_ScriptedModel] = []

        def build(*args: Any, **kwargs: Any) -> _ScriptedModel:
            m = _ScriptedModel(
                [_Reply("", [_tool_call("query_audit_trail")])] * 10
            )
            models.append(m)
            return m

        monkeypatch.setattr(factory, "build_chat_model", build)
        inv.investigate_assessment(
            settings, AuditLogger(), event_id=EVENT_ID,
            assessment_row_id=1, issue_names=("station_agreement",),
            max_rounds=max_rounds,
        )
        assert models[0].invocations == expected_invocations

    def test_unknown_issue_raises(
        self, monkeypatch: pytest.MonkeyPatch, settings: LLMSettings
    ) -> None:
        self._patch_model(monkeypatch, [_Reply("x")])
        with pytest.raises(KeyError):
            inv.investigate_issue(
                settings, AuditLogger(), event_id=EVENT_ID,
                assessment_row_id=1, issue_name="not_an_issue",
            )


class TestInvestigateAssessment:
    def test_investigates_every_issue(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from hazard_assessment.agents.llm_advisory import factory

        monkeypatch.setattr(
            factory, "build_chat_model",
            lambda *a, **k: _ScriptedModel([_Reply("ok")]),
        )
        findings = inv.investigate_assessment(
            LLMSettings(provider="openai", model="m", api_key="k"),
            AuditLogger(), event_id=EVENT_ID, assessment_row_id=5,
        )
        assert [f.issue_name for f in findings] == list(inv.ISSUE_NAMES)

    def test_one_failing_issue_does_not_lose_the_others(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two of three findings serve the operator better than none."""
        from hazard_assessment.agents.llm_advisory import factory

        calls = {"n": 0}

        def flaky(*args: Any, **kwargs: Any) -> _ScriptedModel:
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("provider refused")
            return _ScriptedModel([_Reply("ok")])

        monkeypatch.setattr(factory, "build_chat_model", flaky)
        findings = inv.investigate_assessment(
            LLMSettings(provider="openai", model="m", api_key="k"),
            AuditLogger(), event_id=EVENT_ID, assessment_row_id=5,
        )
        assert len(findings) == len(inv.ISSUE_NAMES) - 1
        assert inv.ISSUE_NAMES[1] not in [f.issue_name for f in findings]
