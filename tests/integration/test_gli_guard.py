"""Integration test for the GLiGuard guardrail.

Skipped cleanly when the optional ``gliner`` extra (``gliner2``) is not installed.
Auto-marked ``e2e`` by the directory conftest.
"""

import pytest

from any_guardrail import AnyGuardrail, GuardrailName
from any_guardrail.types import GuardrailOutput

pytest.importorskip("gliner2", reason="requires `any-guardrail[gliner]`")


def test_gli_guard_benign_text_is_valid() -> None:
    guardrail = AnyGuardrail.create(GuardrailName.GLI_GUARD)
    result = guardrail.validate("What is the weather like today?")
    assert isinstance(result, GuardrailOutput)
    assert result.valid
