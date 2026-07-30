"""Shared pytest configuration.

Isolates the suite from the settings environment. The deployment guide tells
operators to set THRESHOLD_*, INGEST_*, CONFIDENCE_* and LLM_* variables, and
the settings classes in config/settings.py read exactly those prefixes. A shell
with a deployment environment loaded therefore turned assertions about default
values into failures that look like real regressions: THRESHOLD_T1=0.40 alone
fails test_to_threshold_config_default_values, and three INGEST_ variables fail
five tests in test_ingest.py. CI stayed green only because it runs clean.

Tests that need a value set use monkeypatch inside the test body, which runs
after this fixture, so they are unaffected.
"""

from __future__ import annotations

import os

import pytest

_SETTINGS_ENV_PREFIXES = ("THRESHOLD_", "INGEST_", "CONFIDENCE_", "LLM_")


@pytest.fixture(autouse=True)
def _isolate_settings_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove settings variables inherited from the ambient environment."""
    for name in [k for k in os.environ if k.startswith(_SETTINGS_ENV_PREFIXES)]:
        monkeypatch.delenv(name, raising=False)
