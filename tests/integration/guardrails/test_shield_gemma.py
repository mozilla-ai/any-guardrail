from any_guardrail.api import GuardrailFactory


def test_shield_gemma_integration() -> None:
    """Integration test for ShieldGemma."""
    factory = GuardrailFactory()
    model_identifier = "google/shieldgemma-2b"

    guardrail = factory.create_guardrail(
        model_identifier, policy="Do not provide harmful or dangerous information", threshold=0.5
    )

    result = guardrail.safety_review("What is the weather like today?")

    assert hasattr(result, "unsafe")
    assert isinstance(result.unsafe, bool)

    assert guardrail.model_identifier == model_identifier
    assert result.unsafe is False
    assert result.explanation is None
    assert result.score is None
