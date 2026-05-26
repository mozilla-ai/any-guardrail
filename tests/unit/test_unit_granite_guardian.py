from typing import Any
from unittest.mock import MagicMock

import pytest

from any_guardrail import GuardrailOutput
from any_guardrail.guardrails.granite_guardian import GraniteGuardian, GraniteGuardianRisk
from any_guardrail.guardrails.granite_guardian.granite_guardian import (
    GUARDIAN_JUDGE_NOTHINK,
    GUARDIAN_JUDGE_THINK,
    MAX_NEW_TOKENS_NOTHINK,
    MAX_NEW_TOKENS_THINK,
    _parse_generation,
)
from any_guardrail.types import GuardrailInferenceOutput


@pytest.fixture
def guardian_instance() -> GraniteGuardian:
    """Create a GraniteGuardian instance without loading model weights."""
    instance = object.__new__(GraniteGuardian)
    instance.model_id = "ibm-granite/granite-guardian-4.1-8b"
    instance.criteria = GraniteGuardianRisk.HARM
    instance.think = False
    return instance


def _make_mock_provider(generated_text: str = "<score>no</score>") -> MagicMock:
    """Build a provider mock whose ``generate_chat`` returns a uniform inference output."""
    provider = MagicMock()
    provider.generate_chat.return_value = GuardrailInferenceOutput(
        data={
            "generated_text": generated_text,
            "prompt_token_count": 32,
            "completion_token_count": 7,
            "raw": None,
        }
    )
    return provider


@pytest.mark.parametrize(
    ("model_outputs", "expected_valid", "expected_score"),
    [
        ("<score>yes</score>", False, "yes"),
        ("<score>no</score>", True, "no"),
        ("<score>\nyes\n</score>", False, "yes"),
        ("<score>  NO  </score>", True, "no"),
        ("<think>reasoning here</think>\n<score>yes</score>", False, "yes"),
        ("<think>multi\nline\nreasoning</think><score>no</score>", True, "no"),
        ("no score tag at all", None, None),
        ("<score>maybe</score>", None, None),
        ("", None, None),
    ],
)
def test_parse_generation(model_outputs: str, expected_valid: bool | None, expected_score: str | None) -> None:
    """Score parsing handles yes/no, think-stripping, whitespace, case, and malformed output."""
    result = _parse_generation(model_outputs)

    assert isinstance(result, GuardrailOutput)
    assert result.valid == expected_valid
    assert result.score == expected_score
    assert result.explanation == model_outputs


def test_post_processing_extracts_score_from_generated_text(guardian_instance: GraniteGuardian) -> None:
    """Post-processing reads ``generated_text`` and parses the yes/no score."""
    guardian_instance.provider = MagicMock()

    result = guardian_instance._post_processing(
        GuardrailInferenceOutput(
            data={
                "generated_text": "<score>no</score>",
                "prompt_token_count": 10,
                "completion_token_count": 3,
                "raw": None,
            }
        )
    )

    assert result.valid is True
    assert result.score == "no"
    assert result.explanation == "<score>no</score>"


def test_build_guardian_block_nothink(guardian_instance: GraniteGuardian) -> None:
    """No-think mode embeds the no-think instruction and the user's criterion."""
    block = guardian_instance._build_guardian_block()

    assert block.startswith("<guardian>")
    assert GUARDIAN_JUDGE_NOTHINK in block
    assert GUARDIAN_JUDGE_THINK not in block
    assert f"### Criteria: {GraniteGuardianRisk.HARM}" in block
    assert "### Scoring Schema:" in block
    assert "return 'yes'" in block


def test_build_guardian_block_think(guardian_instance: GraniteGuardian) -> None:
    """Think mode switches to the think instruction."""
    guardian_instance.think = True

    block = guardian_instance._build_guardian_block()

    assert GUARDIAN_JUDGE_THINK in block
    assert GUARDIAN_JUDGE_NOTHINK not in block


def test_build_messages_without_output(guardian_instance: GraniteGuardian) -> None:
    """When no assistant response is given, messages are [user, guardian_block]."""
    messages = guardian_instance._build_messages("user input", None)

    assert len(messages) == 2
    assert messages[0] == {"role": "user", "content": "user input"}
    assert messages[1]["role"] == "user"
    assert messages[1]["content"].startswith("<guardian>")


def test_build_messages_with_output(guardian_instance: GraniteGuardian) -> None:
    """When an assistant response is given, it's inserted between user turn and guardian block."""
    messages = guardian_instance._build_messages("user input", "assistant response")

    assert len(messages) == 3
    assert messages[0] == {"role": "user", "content": "user input"}
    assert messages[1] == {"role": "assistant", "content": "assistant response"}
    assert messages[2]["role"] == "user"
    assert messages[2]["content"].startswith("<guardian>")


def test_pre_processing_basic(guardian_instance: GraniteGuardian) -> None:
    """Preprocessing returns the assembled messages and an empty chat_template_kwargs by default."""
    guardian_instance.provider = _make_mock_provider()

    output = guardian_instance._pre_processing("hello")

    assert "messages" in output.data
    assert "chat_template_kwargs" in output.data
    assert output.data["chat_template_kwargs"] == {}
    # The first message is the user input, the last is the guardian block.
    assert output.data["messages"][0] == {"role": "user", "content": "hello"}
    assert output.data["messages"][-1]["content"].startswith("<guardian>")


def test_pre_processing_forwards_documents(guardian_instance: GraniteGuardian) -> None:
    """RAG mode forwards the documents list via chat_template_kwargs."""
    guardian_instance.provider = _make_mock_provider()
    docs = [{"doc_id": "0", "text": "some context"}]

    output = guardian_instance._pre_processing("query", output_text="response", documents=docs)

    assert output.data["chat_template_kwargs"]["documents"] == docs
    assert "available_tools" not in output.data["chat_template_kwargs"]


def test_pre_processing_forwards_available_tools(guardian_instance: GraniteGuardian) -> None:
    """Function-calling mode forwards the tools list via chat_template_kwargs."""
    guardian_instance.provider = _make_mock_provider()
    tools = [{"name": "search", "description": "search the web", "parameters": {}}]

    output = guardian_instance._pre_processing("query", output_text="response", available_tools=tools)

    assert output.data["chat_template_kwargs"]["available_tools"] == tools
    assert "documents" not in output.data["chat_template_kwargs"]


def test_pre_processing_forwards_both_documents_and_tools(guardian_instance: GraniteGuardian) -> None:
    """Agentic RAG can combine documents and tools in a single call."""
    guardian_instance.provider = _make_mock_provider()
    docs = [{"doc_id": "0", "text": "ctx"}]
    tools = [{"name": "tool", "description": "d", "parameters": {}}]

    output = guardian_instance._pre_processing("q", output_text="r", documents=docs, available_tools=tools)

    assert output.data["chat_template_kwargs"]["documents"] == docs
    assert output.data["chat_template_kwargs"]["available_tools"] == tools


def test_pre_processing_assembles_messages_with_output(guardian_instance: GraniteGuardian) -> None:
    """When ``output_text`` is provided, the assistant turn appears between user and guardian block."""
    guardian_instance.provider = _make_mock_provider()

    output = guardian_instance._pre_processing("user q", output_text="assistant a")

    messages = output.data["messages"]
    assert messages[0] == {"role": "user", "content": "user q"}
    assert messages[1] == {"role": "assistant", "content": "assistant a"}
    assert messages[2]["role"] == "user"
    assert "<guardian>" in messages[2]["content"]


def test_inference_passes_max_tokens_per_mode(guardian_instance: GraniteGuardian) -> None:
    """Think mode passes the larger token budget; no-think passes the smaller one."""
    from any_guardrail.types import AnyDict, GuardrailPreprocessOutput

    provider = _make_mock_provider()
    guardian_instance.provider = provider
    inputs: AnyDict = {"messages": [], "chat_template_kwargs": {}}

    guardian_instance._inference(GuardrailPreprocessOutput(data=inputs))
    _, kwargs = provider.generate_chat.call_args
    assert kwargs["max_new_tokens"] == MAX_NEW_TOKENS_NOTHINK
    assert kwargs["do_sample"] is False

    guardian_instance.think = True
    guardian_instance._inference(GuardrailPreprocessOutput(data=inputs))
    _, kwargs = provider.generate_chat.call_args
    assert kwargs["max_new_tokens"] == MAX_NEW_TOKENS_THINK


def test_inference_forwards_chat_template_kwargs(guardian_instance: GraniteGuardian) -> None:
    """``chat_template_kwargs`` from pre-processing reach ``generate_chat`` unchanged when non-empty."""
    provider = _make_mock_provider()
    guardian_instance.provider = provider
    docs = [{"doc_id": "0", "text": "context"}]
    from any_guardrail.types import GuardrailPreprocessOutput

    guardian_instance._inference(
        GuardrailPreprocessOutput(
            data={"messages": [{"role": "user", "content": "x"}], "chat_template_kwargs": {"documents": docs}}
        )
    )

    _, kwargs = provider.generate_chat.call_args
    assert kwargs["chat_template_kwargs"] == {"documents": docs}


def test_inference_passes_none_when_chat_template_kwargs_empty(guardian_instance: GraniteGuardian) -> None:
    """Empty chat_template_kwargs is forwarded as None to keep the provider call shape clean."""
    provider = _make_mock_provider()
    guardian_instance.provider = provider
    from any_guardrail.types import GuardrailPreprocessOutput

    guardian_instance._inference(
        GuardrailPreprocessOutput(data={"messages": [{"role": "user", "content": "x"}], "chat_template_kwargs": {}})
    )

    _, kwargs = provider.generate_chat.call_args
    assert kwargs["chat_template_kwargs"] is None


def test_validate_forwards_documents_and_tools(guardian_instance: GraniteGuardian) -> None:
    """End-to-end validate() forwards documents and tools to ``generate_chat`` via chat_template_kwargs."""
    provider = _make_mock_provider(generated_text="<score>no</score>")
    guardian_instance.provider = provider

    docs = [{"doc_id": "0", "text": "context"}]
    tools = [{"name": "t", "description": "d", "parameters": {}}]

    result = guardian_instance.validate("q", output_text="r", documents=docs, available_tools=tools)

    _, kwargs = provider.generate_chat.call_args
    assert kwargs["chat_template_kwargs"]["documents"] == docs
    assert kwargs["chat_template_kwargs"]["available_tools"] == tools
    assert result.valid is True
    assert result.score == "no"


def test_validate_raises_on_non_chat_capable_provider(guardian_instance: GraniteGuardian) -> None:
    """A provider that doesn't implement ``generate_chat`` surfaces a clear NotImplementedError."""
    from any_guardrail.providers.base import Provider

    class DummyProvider(Provider[Any, Any]):
        def load_model(self, model_id: str, **kwargs: Any) -> None:
            del model_id, kwargs

        def pre_process(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - never reached
            raise NotImplementedError

        def infer(self, model_inputs: Any) -> Any:  # pragma: no cover - never reached
            raise NotImplementedError

    guardian_instance.provider = DummyProvider()

    with pytest.raises(NotImplementedError, match="generate_chat"):
        guardian_instance.validate("test input")


def test_risk_constants_are_nonempty_strings() -> None:
    """All predefined risk constants are non-trivial strings suitable as criteria."""
    risk_names = [
        "HARM",
        "SOCIAL_BIAS",
        "JAILBREAK",
        "VIOLENCE",
        "PROFANITY",
        "UNETHICAL_BEHAVIOR",
        "GROUNDEDNESS",
        "CONTEXT_RELEVANCE",
        "ANSWER_RELEVANCE",
        "FUNCTION_CALL_HALLUCINATION",
    ]
    for name in risk_names:
        value = getattr(GraniteGuardianRisk, name)
        assert isinstance(value, str)
        assert len(value) > 20
