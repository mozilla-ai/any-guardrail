from typing import Any
from unittest.mock import MagicMock, patch

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
    ("model_outputs", "expected_valid", "expected_answer"),
    [
        ("<score>yes</score>", False, "yes"),
        ("<score>no</score>", True, "no"),
        ("<score>\nyes\n</score>", False, "yes"),
        ("<score>  NO  </score>", True, "no"),
        ("<think>reasoning here</think>\n<score>yes</score>", False, "yes"),
        ("<think>multi\nline\nreasoning</think><score>no</score>", True, "no"),
        # Unparsable output fails closed: valid=False with a parse_failure marker.
        ("no score tag at all", False, None),
        ("<score>maybe</score>", False, None),
        ("", False, None),
    ],
)
def test_parse_generation(model_outputs: str, expected_valid: bool, expected_answer: str | None) -> None:
    """Score parsing handles yes/no, think-stripping, whitespace, case, and malformed output."""
    result = _parse_generation(model_outputs, "criteria text")

    assert isinstance(result, GuardrailOutput)
    assert result.valid == expected_valid
    assert result.explanation == model_outputs
    if expected_answer is None:
        assert result.extra == {"parse_failure": True}
        assert result.categories == []
    else:
        assert result.extra is not None
        assert result.extra["raw_answer"] == expected_answer
        assert len(result.categories) == 1
        assert result.categories[0].name == "criteria text"
        assert result.categories[0].triggered is (expected_answer == "yes")


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
    assert result.extra is not None
    assert result.extra["raw_answer"] == "no"
    assert result.explanation == "<score>no</score>"
    # Token counts from generate_chat land in usage.
    assert result.usage is not None
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 3


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
    assert result.extra is not None
    assert result.extra["raw_answer"] == "no"


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


def test_user_supplied_default_hf_provider_loaded_with_causal_lm() -> None:
    """Regression: a default-constructed HuggingFaceProvider must still load as a causal LM.

    Granite Guardian is a causal LM. If a user passes ``HuggingFaceProvider()``
    (whose default ``model_class`` is ``AutoModelForSequenceClassification``),
    GraniteGuardian.__init__ must override the loader for this call so the
    causal-LM head is loaded — not the missing classification head.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from any_guardrail.providers.huggingface import HuggingFaceProvider

    shared_provider = HuggingFaceProvider()
    assert shared_provider.model_class is not AutoModelForCausalLM  # sanity: default is seq-classification

    with patch.object(HuggingFaceProvider, "load_model") as mock_load:
        GraniteGuardian(provider=shared_provider, criteria=GraniteGuardianRisk.HARM)

    mock_load.assert_called_once()
    _, load_kwargs = mock_load.call_args
    assert load_kwargs["model_class"] is AutoModelForCausalLM
    assert load_kwargs["tokenizer_class"] is AutoTokenizer
    # The provider's stored defaults are not mutated.
    assert shared_provider.model_class is not AutoModelForCausalLM


def test_no_provider_path_constructs_provider_with_causal_lm() -> None:
    """When no provider is supplied, GraniteGuardian builds an HF provider configured for causal LM.

    This is the existing default behavior; it should not regress when the
    user-supplied-provider path is hardened.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from any_guardrail.providers.huggingface import HuggingFaceProvider

    with patch.object(HuggingFaceProvider, "load_model"):
        guardian = GraniteGuardian(criteria=GraniteGuardianRisk.HARM)

    assert isinstance(guardian.provider, HuggingFaceProvider)
    assert guardian.provider.model_class is AutoModelForCausalLM
    assert guardian.provider.tokenizer_class is AutoTokenizer


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
