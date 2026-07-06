from typing import Any

import pytest

from any_guardrail import AnyGuardrail, GuardrailName
from any_guardrail.base import GuardrailOutput, ThreeStageGuardrail
from any_guardrail.guardrails.duo_guard.duo_guard import DUOGUARD_CATEGORIES

# All tests in tests/integration/ are auto-marked `e2e` by the directory
# conftest. Individual tests in this module that need a beefy runner are
# additionally marked `heavy` below.


@pytest.mark.parametrize(
    ("guardrail_name", "guardrail_kwargs"),
    [
        (GuardrailName.DEEPSET, {}),
        (GuardrailName.DUOGUARD, {}),
        (GuardrailName.HARMGUARD, {}),
        (GuardrailName.INJECGUARD, {}),
        (GuardrailName.JASPER, {}),
        (GuardrailName.PANGOLIN, {}),
        (GuardrailName.LLAMA_GUARD, {}),
        (GuardrailName.PROTECTAI, {}),
        (GuardrailName.SENTINEL, {}),
        pytest.param(
            GuardrailName.SHIELD_GEMMA,
            {"policy": "Do not provide harmful or dangerous information"},
            marks=pytest.mark.heavy,  # ~5 GB, needs 8 GB+ RAM
        ),
        pytest.param(
            GuardrailName.GLIDER,
            {
                "pass_criteria": "Do not provide harmful or dangerous information",
                "rubric": "0: If information is safe. 1: If information is dangerous",
                # Rubric direction: higher score = more dangerous, so passing means score <= 0.
                "pass_threshold": 0,
                "higher_is_better": False,
            },
            marks=pytest.mark.heavy,  # ~8 GB, needs 10 GB+ RAM
        ),
        pytest.param(
            GuardrailName.GRANITE_GUARDIAN,
            {"criteria": "The text contains harmful or dangerous content."},
            marks=pytest.mark.heavy,  # ~16 GB, needs 20 GB+ RAM
        ),
        # --- issue #179 additions ---
        (GuardrailName.PROMPT_GUARD, {}),  # 86M encoder (Llama license; needs HF auth)
        (GuardrailName.BIELIK_GUARD, {}),  # 0.1B encoder (gated auto; needs HF auth)
        pytest.param(GuardrailName.WILD_GUARD, {}, marks=pytest.mark.heavy),  # 7B
        pytest.param(GuardrailName.NEMOTRON_CONTENT_SAFETY, {}, marks=pytest.mark.heavy),  # 4B
        pytest.param(GuardrailName.POLY_GUARD, {}, marks=pytest.mark.heavy),  # 8B
        pytest.param(GuardrailName.KANANA_SAFEGUARD, {}, marks=pytest.mark.heavy),  # 8B
        pytest.param(
            GuardrailName.DYNA_GUARD,
            {"policy": "1. Do not produce harmful or dangerous content."},
            marks=pytest.mark.heavy,  # 8B
        ),
        pytest.param(
            GuardrailName.GPT_OSS_SAFEGUARD,
            {"policy": "Content is a violation if it is harmful or dangerous."},
            marks=pytest.mark.heavy,  # 20B
        ),
        pytest.param(
            GuardrailName.PROMETHEUS,
            {"rubric": "Score 1: harmful. Score 5: safe.", "pass_threshold": 1},
            marks=pytest.mark.heavy,  # 7B
        ),
        pytest.param(
            GuardrailName.SELENE,
            {"rubric": "Score 1: harmful. Score 5: safe.", "pass_threshold": 1},
            marks=pytest.mark.heavy,  # 8B
        ),
        pytest.param(
            GuardrailName.COMPASS_JUDGER,
            {"criteria": "Is the response safe?", "rubric": "Higher is safer.", "pass_threshold": 1},
            marks=pytest.mark.heavy,  # 7B
        ),
        # --- issue #93 additions ---
        (GuardrailName.QWEN3_GUARD, {}),  # 0.6B decoder
        (GuardrailName.QWEN3_GUARD_STREAM, {}),  # 0.6B, remote-code streaming heads
    ],
)
def test_huggingface_guardrails(guardrail_name: GuardrailName, guardrail_kwargs: dict[str, Any]) -> None:
    """Iterate on all guardrails using the provider pattern."""
    guardrail = AnyGuardrail.create(guardrail_name=guardrail_name, **guardrail_kwargs)

    assert isinstance(guardrail, ThreeStageGuardrail)
    assert hasattr(guardrail, "provider")
    assert guardrail.model_id == (guardrail_kwargs.get("model_id") or guardrail.SUPPORTED_MODELS[0])  # type: ignore[attr-defined]

    result = guardrail.validate("What is the weather like today?")

    assert isinstance(result, GuardrailOutput)
    assert result.valid
    if guardrail_name == GuardrailName.DUOGUARD:
        # The full 12-category probability distribution survives post-processing.
        assert [category.name for category in result.categories] == DUOGUARD_CATEGORIES
        assert all(category.triggered is False for category in result.categories)
        assert all(category.score is not None for category in result.categories)


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
