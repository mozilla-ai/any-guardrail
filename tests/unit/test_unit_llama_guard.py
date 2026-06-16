from unittest.mock import patch

from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    Llama4ForConditionalGeneration,
)

from any_guardrail.guardrails.llama_guard import LlamaGuard
from any_guardrail.guardrails.llama_guard.llama_guard import (
    LLAMA_GUARD_CATEGORIES,
    _parse_violated_categories,
)
from any_guardrail.providers.huggingface import HuggingFaceProvider
from any_guardrail.types import GuardrailInferenceOutput


def test_v3_user_supplied_default_hf_provider_loaded_with_causal_lm() -> None:
    """Llama Guard 3 with a default HuggingFaceProvider must load via ``AutoModelForCausalLM``.

    Otherwise the seq-classification default would silently load the wrong
    head and corrupt downstream chat-template generation.
    """
    shared_provider = HuggingFaceProvider()

    with patch.object(HuggingFaceProvider, "load_model") as mock_load:
        LlamaGuard(model_id="meta-llama/Llama-Guard-3-1B", provider=shared_provider)

    mock_load.assert_called_once()
    _, load_kwargs = mock_load.call_args
    assert load_kwargs["model_class"] is AutoModelForCausalLM
    assert load_kwargs["tokenizer_class"] is AutoTokenizer
    assert shared_provider.model_class is not AutoModelForCausalLM


def test_v4_user_supplied_default_hf_provider_loaded_with_llama4_conditional() -> None:
    """Llama Guard 4 with a default HuggingFaceProvider must load via ``Llama4ForConditionalGeneration``.

    The v4 multimodal variant needs ``AutoProcessor`` (not ``AutoTokenizer``);
    the default seq-classification provider would break both.
    """
    shared_provider = HuggingFaceProvider()

    with patch.object(HuggingFaceProvider, "load_model") as mock_load:
        LlamaGuard(model_id="meta-llama/Llama-Guard-4-12B", provider=shared_provider)

    mock_load.assert_called_once()
    _, load_kwargs = mock_load.call_args
    assert load_kwargs["model_class"] is Llama4ForConditionalGeneration
    assert load_kwargs["tokenizer_class"] is AutoProcessor


def test_v3_no_provider_path_constructs_provider_with_causal_lm() -> None:
    """When no provider is supplied, v3 builds a HuggingFaceProvider with the right causal-LM classes."""
    with patch.object(HuggingFaceProvider, "load_model"):
        guardrail = LlamaGuard(model_id="meta-llama/Llama-Guard-3-1B")

    assert isinstance(guardrail.provider, HuggingFaceProvider)
    assert guardrail.provider.model_class is AutoModelForCausalLM
    assert guardrail.provider.tokenizer_class is AutoTokenizer


def test_v4_no_provider_path_constructs_provider_with_llama4_conditional() -> None:
    """When no provider is supplied, v4 builds a HuggingFaceProvider with ``Llama4ForConditionalGeneration``."""
    with patch.object(HuggingFaceProvider, "load_model"):
        guardrail = LlamaGuard(model_id="meta-llama/Llama-Guard-4-12B")

    assert isinstance(guardrail.provider, HuggingFaceProvider)
    assert guardrail.provider.model_class is Llama4ForConditionalGeneration
    assert guardrail.provider.tokenizer_class is AutoProcessor


def _guardrail_without_weights() -> LlamaGuard:
    instance: LlamaGuard = object.__new__(LlamaGuard)
    instance.model_id = "meta-llama/Llama-Guard-3-1B"
    return instance


def _inference_output(generated_text: str) -> GuardrailInferenceOutput[dict]:  # type: ignore[type-arg]
    return GuardrailInferenceOutput(
        data={"generated_text": generated_text, "prompt_token_count": 12, "completion_token_count": 4, "raw": None}
    )


def test_post_processing_parses_single_s_code_v3_shape() -> None:
    """Llama Guard 3 emits leading newlines before the verdict; the S-code becomes a category."""
    result = _guardrail_without_weights()._post_processing(_inference_output("\n\nunsafe\nS1"))

    assert result.valid is False
    assert [category.name for category in result.categories] == ["S1"]
    assert result.categories[0].description == "Violent Crimes"
    assert result.categories[0].triggered is True
    assert result.explanation == "\n\nunsafe\nS1"


def test_post_processing_parses_multiple_comma_separated_s_codes() -> None:
    result = _guardrail_without_weights()._post_processing(_inference_output("unsafe\nS1,S10,S1"))

    # Deduplicated, order of first appearance, descriptions resolved.
    assert [category.name for category in result.categories] == ["S1", "S10"]
    assert result.categories[1].description == "Hate"


def test_post_processing_keeps_unknown_codes() -> None:
    result = _guardrail_without_weights()._post_processing(_inference_output("unsafe\nS99"))

    assert result.valid is False
    assert [category.name for category in result.categories] == ["S99"]
    assert result.categories[0].description is None


def test_post_processing_safe_output_has_no_categories() -> None:
    result = _guardrail_without_weights()._post_processing(_inference_output("safe"))

    assert result.valid is True
    assert result.categories == []
    # Token counts from generate_chat land in usage.
    assert result.usage is not None
    assert result.usage.prompt_tokens == 12
    assert result.usage.completion_tokens == 4


def test_taxonomy_covers_all_fourteen_categories() -> None:
    assert len(LLAMA_GUARD_CATEGORIES) == 14
    assert _parse_violated_categories("S1, S14")[1].description == "Code Interpreter Abuse"
