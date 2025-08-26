from unittest import mock

from any_guardrail.base import GuardrailOutput
from any_guardrail.guardrails.any_llm.any_llm import DEFAULT_MODEL_ID, AnyLlm


def test_custom_system_prompt() -> None:
    guardrail = AnyLlm()

    with mock.patch(
        "any_guardrail.guardrails.any_llm.any_llm.completion",
        return_value=mock.Mock(
            choices=[mock.Mock(message=mock.Mock(content='{"valid": true, "explanation": "Valid input."}'))]
        ),
    ) as mock_completion:
        result = guardrail.validate(
            "What is the weather like today?",
            policy="Do not provide harmful or dangerous information",
            system_prompt="You are a helpful assistant. Check this policy: {policy}.",
        )

        assert isinstance(result, GuardrailOutput)

        mock_completion.assert_called_once_with(
            model=DEFAULT_MODEL_ID,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Check this policy: Do not provide harmful or dangerous information.",
                },
                {"role": "user", "content": "What is the weather like today?"},
            ],
            response_format=mock.ANY,
        )
