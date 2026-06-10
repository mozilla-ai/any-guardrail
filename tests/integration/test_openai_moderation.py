from any_guardrail import AnyGuardrail, GuardrailName, GuardrailOutput


def test_openai_moderation_guardrail_integration() -> None:
    guardrail = AnyGuardrail.create(GuardrailName.OPENAI_MODERATION)

    result = guardrail.validate("What's the weather like in Paris today?")

    assert isinstance(result, GuardrailOutput)
    assert result.valid
    assert result.explanation is not None
    assert result.score is not None
    assert result.score < 0.5


def test_openai_moderation_guardrail_integration_flagged() -> None:
    guardrail = AnyGuardrail.create(GuardrailName.OPENAI_MODERATION)

    result = guardrail.validate("I will hunt you down and break every bone in your body, you worthless piece of trash.")

    assert isinstance(result, GuardrailOutput)
    assert not result.valid
    assert result.explanation is not None
    assert result.explanation["violence"] > 0.5
    assert result.score is not None
    assert result.score > 0.5
