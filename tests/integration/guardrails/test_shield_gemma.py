from any_guardrail.api import GuardrailFactory

from any_guardrail.guardrails.shield_gemma import ShieldGemma


def test_shield_gemma_integration() -> None:
    """Integration test for ShieldGemma."""
    factory = GuardrailFactory()
    # This is a randomly initialized model of tiny weights that will give random results but is useful for testing that
    # a HF model can be loaded and used.
    model_identifier = "hf-internal-testing/tiny-random-Gemma3ForCausalLM"

    guardrail = factory.create_guardrail(
        model_identifier, policy="Do not provide harmful or dangerous information", threshold=0.5
    )
    assert isinstance(guardrail, ShieldGemma)

    result = guardrail.safety_review("What is the weather like today?")

    assert guardrail.model_identifier == model_identifier
    assert result.unsafe is not None
    assert result.explanation is None
    assert result.score is None
