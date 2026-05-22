from typing import Any

import pytest

from any_guardrail import AnyGuardrail, GuardrailName
from any_guardrail.base import GuardrailOutput, ThreeStageGuardrail
from any_guardrail.guardrails.duo_guard.duo_guard import DUOGUARD_CATEGORIES

# All tests in tests/integration/ are auto-marked `e2e` by the directory
# conftest. Individual tests in this module that need a beefy runner are
# additionally marked `heavy` below.


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
            marks=pytest.mark.heavy,  # ~5 GB, needs 8 GB+ RAM
        ),
        pytest.param(
            GuardrailName.GLIDER,
            {
                "pass_criteria": "Do not provide harmful or dangerous information",
                "rubric": "0: If information is safe. 1: If information is dangerous",
            },
            None,
            marks=pytest.mark.heavy,  # ~8 GB, needs 10 GB+ RAM
        ),
        pytest.param(
            GuardrailName.GRANITE_GUARDIAN,
            {"criteria": "The text contains harmful or dangerous content."},
            None,
            marks=pytest.mark.heavy,  # ~16 GB, needs 20 GB+ RAM
        ),
    ],
)
def test_huggingface_guardrails(
    guardrail_name: GuardrailName, guardrail_kwargs: dict[str, Any], expected_explanation: Any
) -> None:
    """Iterate on all guardrails using the provider pattern."""
    guardrail = AnyGuardrail.create(guardrail_name=guardrail_name, **guardrail_kwargs)

    assert isinstance(guardrail, ThreeStageGuardrail)
    assert hasattr(guardrail, "provider")
    assert guardrail.model_id == (guardrail_kwargs.get("model_id") or guardrail.SUPPORTED_MODELS[0])  # type: ignore[attr-defined]

    result = guardrail.validate("What is the weather like today?")

    assert isinstance(result, GuardrailOutput)
    assert result.valid if guardrail_name != GuardrailName.GLIDER else not result.valid


def test_off_topic_guardrail() -> None:
    """Test off-topic guardrail separately due to its unique behavior."""
    guardrail = AnyGuardrail.create(GuardrailName.OFFTOPIC)

    assert isinstance(guardrail, ThreeStageGuardrail)
    assert hasattr(guardrail, "provider")
    assert guardrail.model_id == "mozilla-ai/jina-embeddings-v2-small-en-off-topic"  # type: ignore[attr-defined]

    result = guardrail.validate(
        "You are a helpful assistant.", comparison_text="Thank you for being a helpful assistant."
    )

    assert isinstance(result, GuardrailOutput)
    assert result.valid
