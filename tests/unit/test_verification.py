"""Unit tests for the VerificationResult schema.

Validates the safety invariants that enforce the ABSTAIN path:
FAIL -> abstain_required, INCOMPLETE -> abstain_required,
PASS does not force abstain_required, abstain_reason required when
abstain_required is True, and the per-check
applicability/prerequisite/result validity table.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from hazard_assessment.schemas.verification import (
    CheckApplicability,
    CheckResult,
    PrerequisiteStatus,
    VerificationCheck,
    VerificationOutcome,
    VerificationResult,
)


def _make_check(
    name: str = "holdout_validation",
    result: CheckResult = CheckResult.PASS,
    evidence: str = "3 of 3 holdout stations within tolerance",
) -> VerificationCheck:
    return VerificationCheck(name=name, result=result, evidence=evidence)


def _make_result(**overrides) -> VerificationResult:
    defaults = {
        "producer": "verification_agent",
        "overall": VerificationOutcome.PASS,
        "checks": [_make_check()],
        "abstain_required": False,
        "abstain_reason": None,
    }
    defaults.update(overrides)
    return VerificationResult(**defaults)


class TestVerificationOutcomePass:
    def test_pass_without_abstain_accepted(self) -> None:
        result = _make_result()
        assert result.overall == VerificationOutcome.PASS
        assert not result.abstain_required

    def test_pass_with_abstain_rejected(self) -> None:
        """PASS + abstain_required=True is contradictory and must be rejected."""
        with pytest.raises(ValidationError, match="abstain_required=True.*PASS"):
            _make_result(
                overall=VerificationOutcome.PASS,
                abstain_required=True,
                abstain_reason="contradictory",
            )


class TestVerificationOutcomePassWithConcerns:
    def test_pass_with_concerns_no_abstain_accepted(self) -> None:
        result = _make_result(
            overall=VerificationOutcome.PASS_WITH_CONCERNS,
            checks=[_make_check(result=CheckResult.CONCERN)],
        )
        assert result.overall == VerificationOutcome.PASS_WITH_CONCERNS

    def test_pass_with_concerns_abstain_rejected(self) -> None:
        with pytest.raises(
            ValidationError,
            match="abstain_required=True.*PASS_WITH_CONCERNS",
        ):
            _make_result(
                overall=VerificationOutcome.PASS_WITH_CONCERNS,
                checks=[_make_check(result=CheckResult.CONCERN)],
                abstain_required=True,
                abstain_reason="concern does not require abstention",
            )


class TestVerificationOutcomeFail:
    def test_fail_with_abstain_accepted(self) -> None:
        result = _make_result(
            overall=VerificationOutcome.FAIL,
            checks=[_make_check(result=CheckResult.FAIL, evidence="RMS > 5cm")],
            abstain_required=True,
            abstain_reason="Verification failed: holdout RMS exceeded tolerance",
        )
        assert result.overall == VerificationOutcome.FAIL
        assert result.abstain_required

    def test_fail_without_abstain_rejected(self) -> None:
        """FAIL must always require ABSTAIN - safety invariant."""
        with pytest.raises(ValidationError, match="abstain_required=False.*FAIL"):
            _make_result(
                overall=VerificationOutcome.FAIL,
                checks=[_make_check(result=CheckResult.FAIL, evidence="failed")],
                abstain_required=False,
            )


class TestVerificationOutcomeIncomplete:
    def test_incomplete_with_abstain_accepted(self) -> None:
        result = _make_result(
            overall=VerificationOutcome.INCOMPLETE,
            checks=[
                VerificationCheck(
                    name="physical_consistency",
                    result=CheckResult.NOT_EVALUATED,
                    evidence="no seismic magnitude available",
                    applicability=CheckApplicability.REQUIRED,
                    prerequisite=PrerequisiteStatus.MISSING,
                )
            ],
            abstain_required=True,
            abstain_reason="required check physical_consistency blocked",
        )
        assert result.overall == VerificationOutcome.INCOMPLETE
        assert result.abstain_required

    def test_incomplete_without_abstain_rejected(self) -> None:
        """INCOMPLETE must always require ABSTAIN - safety invariant."""
        with pytest.raises(
            ValidationError, match="abstain_required=False.*INCOMPLETE"
        ):
            _make_result(
                overall=VerificationOutcome.INCOMPLETE,
                checks=[
                    VerificationCheck(
                        name="physical_consistency",
                        result=CheckResult.NOT_EVALUATED,
                        evidence="no seismic magnitude available",
                        applicability=CheckApplicability.REQUIRED,
                        prerequisite=PrerequisiteStatus.MISSING,
                    )
                ],
                abstain_required=False,
            )

    def test_pass_with_required_fail_is_rejected(self) -> None:
        with pytest.raises(
            ValidationError, match="overall PASS.*derived outcome is FAIL"
        ):
            _make_result(
                overall=VerificationOutcome.PASS,
                checks=[_make_check(result=CheckResult.FAIL)],
                abstain_required=False,
            )

    def test_pass_with_required_missing_is_rejected(self) -> None:
        with pytest.raises(
            ValidationError, match="overall PASS.*derived outcome is INCOMPLETE"
        ):
            _make_result(
                overall=VerificationOutcome.PASS,
                checks=[
                    VerificationCheck(
                        name="data_coverage",
                        result=CheckResult.NOT_EVALUATED,
                        evidence="geometry unavailable",
                        applicability=CheckApplicability.REQUIRED,
                        prerequisite=PrerequisiteStatus.MISSING,
                    )
                ],
                abstain_required=False,
            )


class TestCheckValidityTable:
    """The 4-row applicability/prerequisite/result validity table."""

    def test_default_row_is_required_available(self) -> None:
        check = _make_check()
        assert check.applicability == CheckApplicability.REQUIRED
        assert check.prerequisite == PrerequisiteStatus.AVAILABLE

    def test_not_applicable_row_accepted(self) -> None:
        check = VerificationCheck(
            name="data_coverage",
            result=CheckResult.NOT_APPLICABLE,
            evidence="requirement matrix v1: not applicable at SEISMIC_ONLY",
            applicability=CheckApplicability.NOT_APPLICABLE,
            prerequisite=PrerequisiteStatus.NOT_REQUIRED,
        )
        assert check.result == CheckResult.NOT_APPLICABLE

    def test_missing_prerequisite_requires_not_evaluated(self) -> None:
        with pytest.raises(ValidationError):
            VerificationCheck(
                name="x",
                result=CheckResult.PASS,
                evidence="e",
                prerequisite=PrerequisiteStatus.MISSING,
            )

    def test_not_evaluated_requires_missing_prerequisite(self) -> None:
        with pytest.raises(ValidationError):
            VerificationCheck(
                name="x",
                result=CheckResult.NOT_EVALUATED,
                evidence="e",
            )

    def test_error_result_requires_error_prerequisite(self) -> None:
        with pytest.raises(ValidationError):
            VerificationCheck(
                name="x", result=CheckResult.ERROR, evidence="e"
            )
        ok = VerificationCheck(
            name="x",
            result=CheckResult.ERROR,
            evidence="e",
            prerequisite=PrerequisiteStatus.ERROR,
        )
        assert ok.result == CheckResult.ERROR

    def test_not_applicable_result_requires_na_applicability(self) -> None:
        with pytest.raises(ValidationError):
            VerificationCheck(
                name="x", result=CheckResult.NOT_APPLICABLE, evidence="e"
            )

    def test_na_applicability_rejects_evaluated_result(self) -> None:
        with pytest.raises(ValidationError):
            VerificationCheck(
                name="x",
                result=CheckResult.PASS,
                evidence="e",
                applicability=CheckApplicability.NOT_APPLICABLE,
                prerequisite=PrerequisiteStatus.NOT_REQUIRED,
            )

    def test_not_required_prerequisite_only_with_na(self) -> None:
        with pytest.raises(ValidationError):
            VerificationCheck(
                name="x",
                result=CheckResult.PASS,
                evidence="e",
                prerequisite=PrerequisiteStatus.NOT_REQUIRED,
            )


class TestAbstainReasonRequired:
    def test_abstain_required_without_reason_rejected(self) -> None:
        with pytest.raises(ValidationError, match="abstain_reason is required"):
            _make_result(
                overall=VerificationOutcome.FAIL,
                checks=[_make_check(result=CheckResult.FAIL, evidence="failed")],
                abstain_required=True,
                abstain_reason=None,
            )

    def test_abstain_required_with_empty_reason_rejected(self) -> None:
        with pytest.raises(ValidationError, match="abstain_reason is required"):
            _make_result(
                overall=VerificationOutcome.FAIL,
                checks=[_make_check(result=CheckResult.FAIL, evidence="failed")],
                abstain_required=True,
                abstain_reason="",
            )


class TestChecksValidation:
    def test_at_least_one_check_required(self) -> None:
        with pytest.raises(ValidationError):
            _make_result(checks=[])

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            VerificationCheck(
                name="test", result=CheckResult.PASS,
                evidence="ok", rogue_field="bad",
            )


class TestVerificationResultExtraFields:
    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _make_result(rogue_field="bad")
