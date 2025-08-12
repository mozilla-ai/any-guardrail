from typing import Any

import pytest

from any_guardrail import AnyGuardrail, GuardrailName
from any_guardrail.guardrails.duo_guard import DUOGUARD_CATEGORIES
from any_guardrail.guardrails.huggingface import HuggingFace


@pytest.mark.parametrize(
    ("guardrail_name", "guardrail_kwargs", "expected_explanation"),
    [
        (GuardrailName.DEEPSET, {}, None),
        (GuardrailName.DUOGUARD, {}, dict.fromkeys(DUOGUARD_CATEGORIES, False)),
        (GuardrailName.HARMGUARD, {}, None),
        (GuardrailName.INJECGUARD, {}, None),
        (GuardrailName.JASPER, {}, None),
        (GuardrailName.PANGOLIN, {}, None),
        (GuardrailName.PROTECTAI, {}, None),
        # (GuardrailName.SENTINEL, {}, None),  # Requires HF login
        (GuardrailName.SHIELD_GEMMA, {}, None),
    ],
)
def test_huggingface_guardrails(
    guardrail_name: str, guardrail_kwargs: dict[str, Any], expected_explanation: Any
) -> None:
    """Iterate on all guardrails inheriting from HuggingFace."""
    guardrail = AnyGuardrail.create(guardrail_name=guardrail_name, **guardrail_kwargs)
    assert guardrail.model_id == guardrail.SUPPORTED_MODELS[0]

    assert isinstance(guardrail, HuggingFace)

    result = guardrail.validate("What is the weather like today?")

    assert not result.unsafe
    assert result.explanation == expected_explanation
    assert result.score is not None
