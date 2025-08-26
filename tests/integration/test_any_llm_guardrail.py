from any_guardrail import AnyGuardrail, GuardrailName, GuardrailOutput


def test_any_llm_guardrail() -> None:
    guardrail = AnyGuardrail.create(GuardrailName.ANYLLM)

    result = guardrail.validate(
        "What is the weather like today?", policy="Do not provide harmful or dangerous information"
    )

    assert isinstance(result, GuardrailOutput)

    assert result.valid
    assert result.explanation is not None
