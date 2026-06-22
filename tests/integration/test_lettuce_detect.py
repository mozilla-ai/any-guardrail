"""Integration test for the LettuceDetect guardrail.

Skipped cleanly when the optional ``lettucedetect`` extra is not installed.
Auto-marked ``e2e`` by the directory conftest.
"""

import pytest

from any_guardrail import AnyGuardrail, GuardrailName
from any_guardrail.types import GuardrailOutput

pytest.importorskip("lettucedetect", reason="requires `any-guardrail[lettucedetect]`")


def test_lettuce_detect_grounded_answer_is_valid() -> None:
    guardrail = AnyGuardrail.create(GuardrailName.LETTUCE_DETECT)
    result = guardrail.validate(
        "The capital of France is Paris.",
        context=["France is a country in Europe. Its capital is Paris."],
        question="What is the capital of France?",
    )
    assert isinstance(result, GuardrailOutput)
    assert result.valid
