"""Tests for the AnyGuardrail.evaluate() dispatcher (issue #230)."""

from unittest.mock import MagicMock

import pytest

from any_guardrail import AnyGuardrail, EvaluateArgumentError, GuardrailName
from any_guardrail.evaluate import _BUILDERS

PAIR_OUTPUT_TEXT = [
    GuardrailName.GLIDER,
    GuardrailName.HARMGUARD,
    GuardrailName.DYNA_GUARD,
    GuardrailName.COMPASS_JUDGER,
    GuardrailName.PROMETHEUS,
    GuardrailName.NEMOTRON_CONTENT_SAFETY,
    GuardrailName.KANANA_SAFEGUARD,
    GuardrailName.QWEN3_GUARD,
    GuardrailName.POLY_GUARD,
    GuardrailName.QWEN3_GUARD_STREAM,
    GuardrailName.WILD_GUARD,
    GuardrailName.SELENE,
    GuardrailName.LLAMA_GUARD,
    GuardrailName.GRANITE_GUARDIAN,
    GuardrailName.PATRONUS,
]

# The `_single()` bucket, minus ANYLLM (requires `policy`, tested separately).
SINGLE_NO_REQUIRED_KWARGS = [
    GuardrailName.BEDROCK_GUARDRAILS,
    GuardrailName.DEEPSET,
    GuardrailName.DUOGUARD,
    GuardrailName.INJECGUARD,
    GuardrailName.JASPER,
    GuardrailName.OPENAI_MODERATION,
    GuardrailName.PANGOLIN,
    GuardrailName.PROTECTAI,
    GuardrailName.SENTINEL,
    GuardrailName.SHIELD_GEMMA,
    GuardrailName.PROMPT_GUARD,
    GuardrailName.BIELIK_GUARD,
    GuardrailName.AZURE_CONTENT_SAFETY,
    GuardrailName.WATSONX_GUARDIAN,
    GuardrailName.GLI_GUARD,
    GuardrailName.GPT_OSS_SAFEGUARD,
    GuardrailName.GLI_NER_PII,
    GuardrailName.LAKERA_GUARD,
    GuardrailName.AZURE_PROMPT_SHIELDS,
]

ALL_COVERED = {
    *PAIR_OUTPUT_TEXT,
    *SINGLE_NO_REQUIRED_KWARGS,
    GuardrailName.ALINIA,
    GuardrailName.OFFTOPIC,
    GuardrailName.ANYLLM,
    GuardrailName.LETTUCE_DETECT,
    GuardrailName.FLOWJUDGE,
}


def test_evaluate_covers_all_guardrails_exactly() -> None:
    """Every GuardrailName has exactly one dispatch builder, and vice versa."""
    assert set(_BUILDERS) == set(GuardrailName)


def test_local_guardrail_lists_cover_every_name() -> None:
    """Sanity check that this test file's own groupings haven't drifted from the full set."""
    assert ALL_COVERED == set(GuardrailName)


def _stub() -> MagicMock:
    return MagicMock()


@pytest.mark.parametrize("name", PAIR_OUTPUT_TEXT, ids=lambda n: n.value)
def test_pair_output_text_dispatch_with_response(name: GuardrailName) -> None:
    guardrail = _stub()
    AnyGuardrail.evaluate(name, guardrail, "P", "R")
    guardrail.validate.assert_called_once_with("P", output_text="R")


@pytest.mark.parametrize("name", PAIR_OUTPUT_TEXT, ids=lambda n: n.value)
def test_pair_output_text_dispatch_without_response(name: GuardrailName) -> None:
    guardrail = _stub()
    AnyGuardrail.evaluate(name, guardrail, "P")
    guardrail.validate.assert_called_once_with("P")


def test_granite_guardian_extra_kwargs_pass_through() -> None:
    guardrail = _stub()
    AnyGuardrail.evaluate(GuardrailName.GRANITE_GUARDIAN, guardrail, "P", "R", documents=[{"text": "doc"}])
    guardrail.validate.assert_called_once_with("P", documents=[{"text": "doc"}], output_text="R")


def test_pair_explicit_response_kwarg_wins_over_response() -> None:
    """An explicitly-passed response kwarg takes precedence over the `response` positional."""
    guardrail = _stub()
    AnyGuardrail.evaluate(GuardrailName.PROMETHEUS, guardrail, "P", "ignored", output_text="explicit")
    guardrail.validate.assert_called_once_with("P", output_text="explicit")


def test_alinia_dispatch_uses_output_kwarg() -> None:
    guardrail = _stub()
    AnyGuardrail.evaluate(GuardrailName.ALINIA, guardrail, "P", "R", context_documents=["doc"])
    guardrail.validate.assert_called_once_with("P", context_documents=["doc"], output="R")


def test_alinia_dispatch_without_response() -> None:
    guardrail = _stub()
    AnyGuardrail.evaluate(GuardrailName.ALINIA, guardrail, "P")
    guardrail.validate.assert_called_once_with("P")


def test_offtopic_dispatch_uses_comparison_text_kwarg() -> None:
    guardrail = _stub()
    AnyGuardrail.evaluate(GuardrailName.OFFTOPIC, guardrail, "P", "R")
    guardrail.validate.assert_called_once_with("P", comparison_text="R")


def test_offtopic_missing_response_raises() -> None:
    guardrail = _stub()
    with pytest.raises(EvaluateArgumentError, match="comparison_text"):
        AnyGuardrail.evaluate(GuardrailName.OFFTOPIC, guardrail, "P")
    guardrail.validate.assert_not_called()


@pytest.mark.parametrize("name", SINGLE_NO_REQUIRED_KWARGS, ids=lambda n: n.value)
def test_single_dispatch(name: GuardrailName) -> None:
    guardrail = _stub()
    AnyGuardrail.evaluate(name, guardrail, "P")
    guardrail.validate.assert_called_once_with("P")


@pytest.mark.parametrize("name", SINGLE_NO_REQUIRED_KWARGS, ids=lambda n: n.value)
def test_single_rejects_response(name: GuardrailName) -> None:
    guardrail = _stub()
    with pytest.raises(EvaluateArgumentError, match="response"):
        AnyGuardrail.evaluate(name, guardrail, "P", "R")
    guardrail.validate.assert_not_called()


def test_gli_ner_pii_extra_kwargs_pass_through() -> None:
    guardrail = _stub()
    AnyGuardrail.evaluate(GuardrailName.GLI_NER_PII, guardrail, "P", entity_types=["PERSON"], threshold=0.5)
    guardrail.validate.assert_called_once_with("P", entity_types=["PERSON"], threshold=0.5)


def test_any_llm_dispatch_with_policy() -> None:
    guardrail = _stub()
    AnyGuardrail.evaluate(GuardrailName.ANYLLM, guardrail, "P", policy="Be nice")
    guardrail.validate.assert_called_once_with("P", policy="Be nice")


def test_any_llm_missing_policy_raises() -> None:
    guardrail = _stub()
    with pytest.raises(EvaluateArgumentError, match="policy"):
        AnyGuardrail.evaluate(GuardrailName.ANYLLM, guardrail, "P")
    guardrail.validate.assert_not_called()


def test_lettuce_detect_dispatch_swaps_prompt_and_response_roles() -> None:
    """LettuceDetect's own validate() takes the answer as input_text and the question as a kwarg."""
    guardrail = _stub()
    AnyGuardrail.evaluate(GuardrailName.LETTUCE_DETECT, guardrail, "What is the capital?", "Paris", context="France...")
    guardrail.validate.assert_called_once_with("Paris", question="What is the capital?", context="France...")


def test_lettuce_detect_explicit_question_kwarg_wins_over_prompt() -> None:
    guardrail = _stub()
    AnyGuardrail.evaluate(
        GuardrailName.LETTUCE_DETECT,
        guardrail,
        "ignored prompt",
        "Paris",
        context="France...",
        question="explicit question",
    )
    guardrail.validate.assert_called_once_with("Paris", context="France...", question="explicit question")


def test_lettuce_detect_missing_response_raises() -> None:
    guardrail = _stub()
    with pytest.raises(EvaluateArgumentError, match="response"):
        AnyGuardrail.evaluate(GuardrailName.LETTUCE_DETECT, guardrail, "What is the capital?", context="France...")
    guardrail.validate.assert_not_called()


def test_lettuce_detect_missing_context_raises() -> None:
    guardrail = _stub()
    with pytest.raises(EvaluateArgumentError, match="context"):
        AnyGuardrail.evaluate(GuardrailName.LETTUCE_DETECT, guardrail, "What is the capital?", "Paris")
    guardrail.validate.assert_not_called()


def test_flowjudge_dispatch_builds_output_from_response() -> None:
    guardrail = _stub()
    guardrail.required_output = "response"
    AnyGuardrail.evaluate(GuardrailName.FLOWJUDGE, guardrail, "unused prompt", "The answer", inputs=[{"query": "Q"}])
    guardrail.validate.assert_called_once_with([{"query": "Q"}], output={"response": "The answer"})


def test_flowjudge_missing_inputs_raises() -> None:
    guardrail = _stub()
    guardrail.required_output = "response"
    with pytest.raises(EvaluateArgumentError, match="inputs"):
        AnyGuardrail.evaluate(GuardrailName.FLOWJUDGE, guardrail, "P", "R")
    guardrail.validate.assert_not_called()


def test_flowjudge_missing_output_raises_when_no_response() -> None:
    guardrail = _stub()
    guardrail.required_output = "response"
    with pytest.raises(EvaluateArgumentError, match="output"):
        AnyGuardrail.evaluate(GuardrailName.FLOWJUDGE, guardrail, "P", inputs=[{"query": "Q"}])
    guardrail.validate.assert_not_called()


def test_flowjudge_explicit_output_kwarg_wins_over_response() -> None:
    guardrail = _stub()
    guardrail.required_output = "response"
    AnyGuardrail.evaluate(
        GuardrailName.FLOWJUDGE,
        guardrail,
        "P",
        "ignored",
        inputs=[{"query": "Q"}],
        output={"response": "explicit"},
    )
    guardrail.validate.assert_called_once_with([{"query": "Q"}], output={"response": "explicit"})
