from any_guardrail.api import GuardrailFactory
from unittest.mock import patch

from any_guardrail.guardrails.shield_gemma import ShieldGemma


def test_shield_gemma_integration() -> None:
    """Integration test for ShieldGemma."""
    factory = GuardrailFactory()
    # This is a randomly initialized model of tiny weights that will give random results but is useful for testing that
    # a HF model can be loaded and used. Just need to mock the tokenizer because shield_gemma is accessing 'Yes' and 'No' tokens
    model_identifier = "hf-internal-testing/tiny-random-Gemma3ForCausalLM"

    mock_vocab = {"Yes": 100, "No": 101}

    guardrail = factory.create_guardrail(
        model_identifier, policy="Do not provide harmful or dangerous information", threshold=0.5
    )
    assert isinstance(guardrail, ShieldGemma)

    # Patch the get_vocab method on the tokenizer instance
    with patch.object(guardrail.guardrail.tokenizer, "get_vocab", return_value=mock_vocab):
        result = guardrail.safety_review("What is the weather like today?")

        assert guardrail.model_identifier == model_identifier
        assert result.unsafe is not None
        assert result.explanation is None
        assert result.score is None
