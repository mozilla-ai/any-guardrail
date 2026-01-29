import os
from typing import Any

import pytest

from any_guardrail import AnyGuardrail, GuardrailName
from any_guardrail.base import GuardrailOutput
from any_guardrail.guardrails.duo_guard.duo_guard import DUOGUARD_CATEGORIES
from any_guardrail.guardrails.huggingface import HuggingFace

RUNNING_IN_CI = os.environ.get("CI") == "true"


@pytest.mark.parametrize(
    ("guardrail_name", "guardrail_kwargs", "expected_explanation"),
    [
        (GuardrailName.DEEPSET, {}, None),
        (GuardrailName.DUOGUARD, {}, dict.fromkeys(DUOGUARD_CATEGORIES, False)),
        (GuardrailName.HARMGUARD, {}, None),
        (GuardrailName.INJECGUARD, {}, None),
        (GuardrailName.JASPER, {}, None),
        (GuardrailName.PANGOLIN, {}, None),
        (GuardrailName.LLAMA_GUARD, {}, None),
        (GuardrailName.PROTECTAI, {}, None),
        (GuardrailName.SENTINEL, {}, None),
        pytest.param(
            GuardrailName.SHIELD_GEMMA,
            {"policy": "Do not provide harmful or dangerous information"},
            None,
            marks=pytest.mark.skipif(RUNNING_IN_CI, reason="Model too large for CI runners (~5GB, needs 8GB+ RAM)"),
        ),
        pytest.param(
            GuardrailName.GLIDER,
            {
                "pass_criteria": "Do not provide harmful or dangerous information",
                "rubric": "0: If information is safe. 1: If information is dangerous",
            },
            None,
            marks=pytest.mark.skipif(RUNNING_IN_CI, reason="Model too large for CI runners (~8GB, needs 10GB+ RAM)"),
        ),
    ],
)
def test_huggingface_guardrails(
    guardrail_name: GuardrailName, guardrail_kwargs: dict[str, Any], expected_explanation: Any
) -> None:
    """Iterate on all guardrails inheriting from HuggingFace."""
    guardrail = AnyGuardrail.create(guardrail_name=guardrail_name, **guardrail_kwargs)

    assert isinstance(guardrail, HuggingFace)
    assert guardrail.model_id == (guardrail_kwargs.get("model_id") or guardrail.SUPPORTED_MODELS[0])

    result = guardrail.validate("What is the weather like today?")

    assert isinstance(result, GuardrailOutput)
    assert result.valid if guardrail_name != GuardrailName.GLIDER else not result.valid


def test_off_topic_guardrail() -> None:
    """Test off-topic guardrail separately due to its unique behavior."""
    guardrail = AnyGuardrail.create(GuardrailName.OFFTOPIC)

    assert isinstance(guardrail, HuggingFace)
    assert guardrail.model_id == "mozilla-ai/jina-embeddings-v2-small-en-off-topic"

    result = guardrail.validate("You are a helpful assistant.", "Thank you for being a helpful assistant.")  # type: ignore[call-arg]

    assert isinstance(result, GuardrailOutput)
    assert result.valid
