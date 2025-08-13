from any_guardrail import AnyGuardrail
from any_guardrail.guardrail import GuardrailName
from any_guardrail.types import GuardrailOutput


def test_any_llm_guardrail() -> None:
    guardrail = AnyGuardrail.create(GuardrailName.ANYLLM)

    result = guardrail.validate(
        "What is the weather like today?", policy="Do not provide harmful or dangerous information"
    )

    assert isinstance(result, GuardrailOutput)

    assert not result.unsafe
    assert result.explanation is not None
