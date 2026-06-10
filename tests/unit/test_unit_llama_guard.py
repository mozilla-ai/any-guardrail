from unittest.mock import patch

from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    Llama4ForConditionalGeneration,
)

from any_guardrail.guardrails.llama_guard import LlamaGuard
from any_guardrail.providers.huggingface import HuggingFaceProvider


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
