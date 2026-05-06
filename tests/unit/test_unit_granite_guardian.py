from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from any_guardrail import GuardrailOutput
from any_guardrail.guardrails.granite_guardian import GraniteGuardian, GraniteGuardianRisk
from any_guardrail.guardrails.granite_guardian.granite_guardian import (
    GUARDIAN_JUDGE_NOTHINK,
    GUARDIAN_JUDGE_THINK,
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
    instance._cached_input_length = 0
    return instance


def _make_mock_provider(apply_chat_template_return: dict[str, Any]) -> MagicMock:
    provider = MagicMock()
    provider.tokenizer.apply_chat_template.return_value = apply_chat_template_return
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


def test_post_processing_decodes_generated_suffix(guardian_instance: GraniteGuardian) -> None:
    """Post-processing slices off the prompt tokens and decodes only the generated tail."""
    guardian_instance._cached_input_length = 10
    full_tokens = torch.tensor([[0] * 10 + [1, 2, 3]])

    provider = MagicMock()
    provider.tokenizer.decode.return_value = "<score>no</score>"
    guardian_instance.provider = provider

    result = guardian_instance._post_processing(GuardrailInferenceOutput(data=full_tokens))

    # Verify only the generated suffix was passed to the decoder.
    call_args = provider.tokenizer.decode.call_args
    passed_tokens = call_args.args[0]
    assert passed_tokens.tolist() == [1, 2, 3]
    assert call_args.kwargs["skip_special_tokens"] is True
    assert result.valid is True
    assert result.score == "no"


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
    """Preprocessing invokes apply_chat_template with the expected template kwargs."""
    fake_tokens = {"input_ids": torch.tensor([[1, 2, 3, 4]]), "attention_mask": torch.tensor([[1, 1, 1, 1]])}
    guardian_instance.provider = _make_mock_provider(fake_tokens)

    output = guardian_instance._pre_processing("hello")

    tokenizer = guardian_instance.provider.tokenizer
    tokenizer.apply_chat_template.assert_called_once()
    _, kwargs = tokenizer.apply_chat_template.call_args
    assert kwargs["add_generation_prompt"] is True
    assert kwargs["tokenize"] is True
    assert kwargs["return_dict"] is True
    assert kwargs["return_tensors"] == "pt"
    assert "documents" not in kwargs
    assert "available_tools" not in kwargs
    assert guardian_instance._cached_input_length == 4
    assert output.data is fake_tokens


def test_pre_processing_forwards_documents(guardian_instance: GraniteGuardian) -> None:
    """RAG mode forwards the documents list to apply_chat_template."""
    fake_tokens = {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}
    guardian_instance.provider = _make_mock_provider(fake_tokens)
    docs = [{"doc_id": "0", "text": "some context"}]

    guardian_instance._pre_processing("query", output_text="response", documents=docs)

    _, kwargs = guardian_instance.provider.tokenizer.apply_chat_template.call_args
    assert kwargs["documents"] == docs
    assert "available_tools" not in kwargs


def test_pre_processing_forwards_available_tools(guardian_instance: GraniteGuardian) -> None:
    """Function-calling mode forwards the tools list to apply_chat_template."""
    fake_tokens = {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.tensor([[1, 1]])}
    guardian_instance.provider = _make_mock_provider(fake_tokens)
    tools = [{"name": "search", "description": "search the web", "parameters": {}}]

    guardian_instance._pre_processing("query", output_text="response", available_tools=tools)

    _, kwargs = guardian_instance.provider.tokenizer.apply_chat_template.call_args
    assert kwargs["available_tools"] == tools
    assert "documents" not in kwargs


def test_pre_processing_forwards_both_documents_and_tools(guardian_instance: GraniteGuardian) -> None:
    """Agentic RAG can combine documents and tools in a single call."""
    fake_tokens = {"input_ids": torch.tensor([[1]]), "attention_mask": torch.tensor([[1]])}
    guardian_instance.provider = _make_mock_provider(fake_tokens)
    docs = [{"doc_id": "0", "text": "ctx"}]
    tools = [{"name": "tool", "description": "d", "parameters": {}}]

    guardian_instance._pre_processing("q", output_text="r", documents=docs, available_tools=tools)

    _, kwargs = guardian_instance.provider.tokenizer.apply_chat_template.call_args
    assert kwargs["documents"] == docs
    assert kwargs["available_tools"] == tools


def test_pre_processing_passes_messages_positionally(guardian_instance: GraniteGuardian) -> None:
    """The messages list is the first positional argument to apply_chat_template."""
    fake_tokens = {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.tensor([[1, 1]])}
    guardian_instance.provider = _make_mock_provider(fake_tokens)

    guardian_instance._pre_processing("user q", output_text="assistant a")

    args, _ = guardian_instance.provider.tokenizer.apply_chat_template.call_args
    messages = args[0]
    assert messages[0] == {"role": "user", "content": "user q"}
    assert messages[1] == {"role": "assistant", "content": "assistant a"}
    assert messages[2]["role"] == "user"
    assert "<guardian>" in messages[2]["content"]


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
