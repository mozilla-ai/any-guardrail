from any_guardrail import AnyGuardrail, GuardrailName
from any_guardrail.guardrails.shield_gemma import ShieldGemma


def test_shield_gemma_integration() -> None:
    """Integration test for ShieldGemma."""
    # This is a randomly initialized model of tiny weights that will give random results but is useful for testing that
    # a HF model can be loaded and used.
    model_id = "hf-internal-testing/tiny-random-Gemma3ForCausalLM"

    guardrail = AnyGuardrail.create(
        model_id=model_id,
        guardrail_name=GuardrailName.SHIELD_GEMMA,
        policy="Do not provide harmful or dangerous information",
        threshold=0.5,
    )
    assert isinstance(guardrail, ShieldGemma)

    result = guardrail.validate("What is the weather like today?")

    assert guardrail.model_id == model_id
    assert result.unsafe is not None
    assert result.explanation is None
    assert result.score is None
