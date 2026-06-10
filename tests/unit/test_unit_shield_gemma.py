from unittest.mock import patch

from transformers import AutoModelForCausalLM, AutoTokenizer

from any_guardrail.guardrails.shield_gemma import ShieldGemma
from any_guardrail.providers.huggingface import HuggingFaceProvider


def test_user_supplied_default_hf_provider_loaded_with_causal_lm() -> None:
    """Regression: a default-constructed HuggingFaceProvider must still load as a causal LM.

    A user app that shares one ``HuggingFaceProvider()`` (default
    ``model_class=AutoModelForSequenceClassification``) across multiple
    guardrails used to silently load ShieldGemma as
    ``Gemma2ForSequenceClassification`` — the checkpoint has no
    ``score.weight``, so it was randomly initialized and ``_post_processing``
    later crashed with ``IndexError: too many indices for array`` (2D logits
    instead of the expected 3D causal-LM logits).

    ShieldGemma must now override the loader for this load so the causal-LM
    head is loaded, without mutating the provider's stored defaults.
    """
    shared_provider = HuggingFaceProvider()
    assert shared_provider.model_class is not AutoModelForCausalLM  # sanity: default is seq-classification

    with patch.object(HuggingFaceProvider, "load_model") as mock_load:
        ShieldGemma(policy="Do not provide harmful content", provider=shared_provider)

    mock_load.assert_called_once()
    _, load_kwargs = mock_load.call_args
    assert load_kwargs["model_class"] is AutoModelForCausalLM
    assert load_kwargs["tokenizer_class"] is AutoTokenizer
    # The provider's stored defaults are not mutated.
    assert shared_provider.model_class is not AutoModelForCausalLM


def test_no_provider_path_constructs_provider_with_causal_lm() -> None:
    """When no provider is supplied, ShieldGemma builds an HF provider configured for causal LM."""
    with patch.object(HuggingFaceProvider, "load_model"):
        guardrail = ShieldGemma(policy="Do not provide harmful content")

    assert isinstance(guardrail.provider, HuggingFaceProvider)
    assert guardrail.provider.model_class is AutoModelForCausalLM
    assert guardrail.provider.tokenizer_class is AutoTokenizer
