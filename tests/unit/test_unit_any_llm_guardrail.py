from unittest import mock

from any_guardrail.base import GuardrailOutput
from any_guardrail.guardrails.any_llm.any_llm import DEFAULT_MODEL_ID, AnyLlm


def test_custom_system_prompt() -> None:
    guardrail = AnyLlm()

    with mock.patch(
        "any_guardrail.guardrails.any_llm.any_llm.completion",
        return_value=mock.Mock(
            choices=[
                mock.Mock(
                    message=mock.Mock(content='{"valid": true, "explanation": "Valid input.", "risk_score": 0.1}')
                )
            ]
        ),
    ) as mock_completion:
        result = guardrail.validate(
            "What is the weather like today?",
            policy="Do not provide harmful or dangerous information",
            system_prompt="You are a helpful assistant. Check this policy: {policy}.",
        )

        assert isinstance(result, GuardrailOutput)
        assert result.valid is True
        assert result.explanation == "Valid input."
        assert result.score == 0.1
        assert result.usage is not None
        assert result.usage.model_id == DEFAULT_MODEL_ID
        assert result.usage.latency_ms is not None

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


def test_unparsable_llm_response_fails_closed() -> None:
    guardrail = AnyLlm()

    with mock.patch(
        "any_guardrail.guardrails.any_llm.any_llm.completion",
        return_value=mock.Mock(choices=[mock.Mock(message=mock.Mock(content="not json at all"))]),
    ):
        result = guardrail.validate(
            "What is the weather like today?",
            policy="Do not provide harmful or dangerous information",
        )

    assert result.valid is False
    assert result.extra == {"parse_failure": True}
    assert result.explanation == "not json at all"


def test_explicit_model_id_is_forwarded_unchanged() -> None:
    """A caller-supplied ``model_id`` wins over the default and is never remapped.

    The default moved off ``openai:gpt-5-nano`` (shut down by OpenAI on 2026-12-11), but a caller who
    pinned that id keeps exactly what they asked for instead of being silently switched to another judge.
    """
    guardrail = AnyLlm()
    pinned = "openai:gpt-5-nano"

    with mock.patch(
        "any_guardrail.guardrails.any_llm.any_llm.completion",
        return_value=mock.Mock(
            choices=[
                mock.Mock(
                    message=mock.Mock(content='{"valid": true, "explanation": "Valid input.", "risk_score": 0.0}')
                )
            ]
        ),
    ) as mock_completion:
        result = guardrail.validate("hello", policy="Be polite.", model_id=pinned)

    assert pinned != DEFAULT_MODEL_ID
    assert mock_completion.call_args.kwargs["model"] == pinned
    assert result.usage is not None
    assert result.usage.model_id == pinned


def test_default_model_id_is_documented() -> None:
    """The docstrings that advertise the default judge name the current ``DEFAULT_MODEL_ID``.

    The generated parameter data, schema and API page are built from these docstrings, so a default that is
    bumped without updating them would publish a stale model name that no generator check notices.
    """
    assert DEFAULT_MODEL_ID in (AnyLlm.__doc__ or "")
    assert DEFAULT_MODEL_ID in (AnyLlm.validate.__doc__ or "")
