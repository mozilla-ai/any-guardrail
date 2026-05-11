from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from any_guardrail import GuardrailOutput
from any_guardrail.guardrails.deepset.deepset import DEEPSET_INJECTION_LABEL, Deepset
from any_guardrail.guardrails.protectai.protectai import PROTECTAI_INJECTION_LABEL, Protectai
from any_guardrail.guardrails.utils import match_injection_label, match_injection_label_batch
from any_guardrail.providers.huggingface import HuggingFaceProvider
from any_guardrail.types import GuardrailInferenceOutput


@pytest.fixture
def protectai_instance() -> Protectai:
    """Build a Protectai instance with a mocked provider."""
    instance = object.__new__(Protectai)
    provider = MagicMock()
    provider.model.config.id2label = {0: "SAFE", 1: PROTECTAI_INJECTION_LABEL}
    instance.provider = provider
    return instance


@pytest.fixture
def deepset_instance() -> Deepset:
    """Build a Deepset instance with a mocked provider."""
    instance = object.__new__(Deepset)
    provider = MagicMock()
    provider.model.config.id2label = {0: "SAFE", 1: DEEPSET_INJECTION_LABEL}
    instance.provider = provider
    return instance


def test_pre_process_list_adds_padding() -> None:
    """When given a list of texts, pre_process passes padding=True to the tokenizer."""
    tokenizer = MagicMock(return_value={"input_ids": torch.tensor([[1, 2], [3, 4]])})

    provider = HuggingFaceProvider()
    provider.tokenizer = tokenizer

    provider.pre_process(["hello", "world"])

    _, kwargs = tokenizer.call_args
    assert kwargs["padding"] is True
    assert kwargs["return_tensors"] == "pt"


def test_pre_process_list_respects_existing_padding_kwarg() -> None:
    """When caller specifies padding, pre_process does not override it for lists."""
    tokenizer = MagicMock(return_value={"input_ids": torch.tensor([[1, 2], [3, 4]])})

    provider = HuggingFaceProvider()
    provider.tokenizer = tokenizer

    provider.pre_process(["hello", "world"], padding="max_length")

    _, kwargs = tokenizer.call_args
    assert kwargs["padding"] == "max_length"


def test_pre_process_str_does_not_add_padding() -> None:
    """A single string input is not auto-padded."""
    tokenizer = MagicMock(return_value={"input_ids": torch.tensor([[1, 2, 3]])})

    provider = HuggingFaceProvider()
    provider.tokenizer = tokenizer

    provider.pre_process("hello")

    _, kwargs = tokenizer.call_args
    assert "padding" not in kwargs


def test_match_injection_label_batch_returns_list() -> None:
    """The batch helper returns one GuardrailOutput per row of logits."""
    # 2 inputs, 2 classes. Row 0 -> class 0 (SAFE), Row 1 -> class 1 (INJECTION).
    logits = torch.tensor([[5.0, 0.0], [0.0, 5.0]])
    outputs = match_injection_label_batch(
        GuardrailInferenceOutput(data={"logits": logits}),
        injection_label="INJECTION",
        id2label={0: "SAFE", 1: "INJECTION"},
    )

    assert len(outputs) == 2
    assert outputs[0].valid is True
    assert outputs[1].valid is False
    assert outputs[0].score is not None
    assert outputs[1].score is not None
    assert outputs[0].score > 0.99
    assert outputs[1].score > 0.99


def test_match_injection_label_single_unchanged() -> None:
    """The single-input helper still returns a single GuardrailOutput from a 1-row batch."""
    logits = torch.tensor([[5.0, 0.0]])
    result = match_injection_label(
        GuardrailInferenceOutput(data={"logits": logits}),
        injection_label="INJECTION",
        id2label={0: "SAFE", 1: "INJECTION"},
    )

    assert isinstance(result, GuardrailOutput)
    assert result.valid is True


def test_validate_batch_protectai_uses_batch_path(protectai_instance: Protectai) -> None:
    """Calling validate with a list dispatches to _validate_batch on a standard guardrail."""
    logits = torch.tensor([[5.0, 0.0], [0.0, 5.0]])
    inference_output = GuardrailInferenceOutput(data={"logits": logits})

    pre_output = MagicMock()
    protectai_instance.provider.pre_process.return_value = pre_output  # type: ignore[attr-defined]
    protectai_instance.provider.infer.return_value = inference_output  # type: ignore[attr-defined]

    results = protectai_instance.validate(["safe text", "injection text"])

    assert isinstance(results, list)
    assert len(results) == 2
    assert results[0].valid is True
    assert results[1].valid is False
    # pre_process gets the list directly (so the tokenizer can batch with padding).
    protectai_instance.provider.pre_process.assert_called_once_with(["safe text", "injection text"])  # type: ignore[attr-defined]
    # infer is called once for the entire batch.
    protectai_instance.provider.infer.assert_called_once_with(pre_output)  # type: ignore[attr-defined]


def test_validate_single_string_returns_single_output(protectai_instance: Protectai) -> None:
    """Single-string validate keeps its existing single-output behavior."""
    logits = torch.tensor([[5.0, 0.0]])
    inference_output = GuardrailInferenceOutput(data={"logits": logits})

    pre_output = MagicMock()
    protectai_instance.provider.pre_process.return_value = pre_output  # type: ignore[attr-defined]
    protectai_instance.provider.infer.return_value = inference_output  # type: ignore[attr-defined]

    result = protectai_instance.validate("hello")

    assert isinstance(result, GuardrailOutput)
    assert result.valid is True


def test_validate_batch_default_iterates() -> None:
    """A guardrail that doesn't override _validate_batch falls back to per-item iteration."""
    from any_guardrail.base import ThreeStageGuardrail
    from any_guardrail.types import GuardrailPreprocessOutput

    class FakeGuardrail(ThreeStageGuardrail[dict, dict, bool, None, float]):  # type: ignore[type-arg]
        def __init__(self) -> None:
            self.calls: list[str] = []

        def _pre_processing(self, input_text, **kwargs):  # type: ignore[no-untyped-def]
            self.calls.append(input_text)
            return GuardrailPreprocessOutput(data={"text": input_text})

        def _inference(self, model_inputs):  # type: ignore[no-untyped-def]
            return GuardrailInferenceOutput(data=model_inputs.data)

        def _post_processing(self, model_outputs):  # type: ignore[no-untyped-def]
            return GuardrailOutput(valid=True, score=1.0)

    guardrail = FakeGuardrail()
    results = guardrail.validate(["a", "b", "c"])

    assert isinstance(results, list)
    assert len(results) == 3
    # The default _validate_batch goes through validate() per item, hitting _pre_processing each time.
    assert guardrail.calls == ["a", "b", "c"]


def test_match_injection_label_batch_uses_softmax_per_row() -> None:
    """Softmax is applied independently per row so scores are well-formed probabilities."""
    # Each row's max softmax score should be close to 1 / (1 + exp(-diff)).
    logits = torch.tensor([[2.0, 1.0], [1.0, 3.0]])
    outputs = match_injection_label_batch(
        GuardrailInferenceOutput(data={"logits": logits}),
        injection_label="INJECTION",
        id2label={0: "SAFE", 1: "INJECTION"},
    )

    expected_row0 = float(np.exp(2.0) / (np.exp(2.0) + np.exp(1.0)))
    expected_row1 = float(np.exp(3.0) / (np.exp(1.0) + np.exp(3.0)))
    assert outputs[0].score == pytest.approx(expected_row0)
    assert outputs[1].score == pytest.approx(expected_row1)
