from unittest.mock import MagicMock

import numpy as np
import pytest

from any_guardrail import GuardrailOutput
from any_guardrail.guardrails.deepset.deepset import Deepset
from any_guardrail.guardrails.protectai.protectai import PROTECTAI_INJECTION_LABEL, Protectai
from any_guardrail.guardrails.utils import match_injection_label, match_injection_label_batch
from any_guardrail.providers.huggingface import HuggingFaceProvider
from any_guardrail.types import AnyDict, GuardrailInferenceOutput


@pytest.fixture
def protectai_instance() -> Protectai:
    """Build a Protectai instance with a mocked provider."""
    instance = object.__new__(Protectai)
    instance.provider = MagicMock()
    return instance


@pytest.fixture
def deepset_instance() -> Deepset:
    """Build a Deepset instance with a mocked provider."""
    instance = object.__new__(Deepset)
    instance.provider = MagicMock()
    return instance


def _uniform_output(predicted_labels: list[str], scores: np.ndarray) -> GuardrailInferenceOutput[AnyDict]:
    """Build an inference output in the uniform shape the providers return."""
    return GuardrailInferenceOutput(
        data={
            "logits": scores,  # logits aren't used downstream for these guardrails
            "scores": scores,
            "predicted_indices": [int(np.argmax(row)) for row in scores],
            "predicted_labels": predicted_labels,
        }
    )


def test_pre_process_list_adds_padding() -> None:
    """When given a list of texts, pre_process passes padding=True to the tokenizer."""
    tokenizer = MagicMock(return_value={"input_ids": [[1, 2], [3, 4]]})

    provider = HuggingFaceProvider()
    provider.tokenizer = tokenizer

    provider.pre_process(["hello", "world"])

    _, kwargs = tokenizer.call_args
    assert kwargs["padding"] is True
    assert kwargs["return_tensors"] == "pt"


def test_pre_process_list_respects_existing_padding_kwarg() -> None:
    """When caller specifies padding, pre_process does not override it for lists."""
    tokenizer = MagicMock(return_value={"input_ids": [[1, 2], [3, 4]]})

    provider = HuggingFaceProvider()
    provider.tokenizer = tokenizer

    provider.pre_process(["hello", "world"], padding="max_length")

    _, kwargs = tokenizer.call_args
    assert kwargs["padding"] == "max_length"


def test_pre_process_str_does_not_add_padding() -> None:
    """A single string input is not auto-padded."""
    tokenizer = MagicMock(return_value={"input_ids": [[1, 2, 3]]})

    provider = HuggingFaceProvider()
    provider.tokenizer = tokenizer

    provider.pre_process("hello")

    _, kwargs = tokenizer.call_args
    assert "padding" not in kwargs


def test_match_injection_label_batch_returns_list() -> None:
    """The batch helper returns one GuardrailOutput per row of scores."""
    # 2 inputs, 2 classes. Row 0 -> SAFE, Row 1 -> INJECTION.
    scores = np.array([[0.99, 0.01], [0.01, 0.99]])
    outputs = match_injection_label_batch(
        _uniform_output(["SAFE", "INJECTION"], scores),
        injection_label="INJECTION",
    )

    assert len(outputs) == 2
    assert outputs[0].valid is True
    assert outputs[1].valid is False
    # score is canonically P(injection): low for the safe row, high for the unsafe row.
    assert outputs[0].score is not None
    assert outputs[1].score is not None
    assert outputs[0].score < 0.02
    assert outputs[1].score > 0.98


def test_match_injection_label_single_unchanged() -> None:
    """The single-input helper returns a single GuardrailOutput from a 1-row batch."""
    scores = np.array([[0.99, 0.01]])
    result = match_injection_label(
        _uniform_output(["SAFE"], scores),
        injection_label="INJECTION",
    )

    assert isinstance(result, GuardrailOutput)
    assert result.valid is True
    assert result.score == pytest.approx(0.01)


def test_validate_batch_protectai_uses_batch_path(protectai_instance: Protectai) -> None:
    """Calling validate with a list dispatches to _validate_batch on a standard guardrail."""
    scores = np.array([[0.99, 0.01], [0.01, 0.99]])
    inference_output = _uniform_output(["SAFE", PROTECTAI_INJECTION_LABEL], scores)

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
    scores = np.array([[0.99, 0.01]])
    inference_output = _uniform_output(["SAFE"], scores)

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

    class FakeGuardrail(ThreeStageGuardrail[dict, dict]):  # type: ignore[type-arg]
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


def test_match_injection_label_batch_reads_score_per_row() -> None:
    """Scores are read per-row from the uniform output shape and report P(injection)."""
    scores = np.array([[0.73, 0.27], [0.12, 0.88]])
    outputs = match_injection_label_batch(
        _uniform_output(["SAFE", "INJECTION"], scores),
        injection_label="INJECTION",
    )

    assert outputs[0].score == pytest.approx(0.27)
    assert outputs[1].score == pytest.approx(0.88)


def test_match_injection_label_categories_with_full_label_list() -> None:
    """When the provider surfaces `labels`, categories carry the full distribution."""
    scores = np.array([[0.73, 0.27]])
    inference_output = _uniform_output(["SAFE"], scores)
    inference_output.data["labels"] = ["SAFE", "INJECTION"]

    result = match_injection_label(inference_output, injection_label="INJECTION")

    assert [category.name for category in result.categories] == ["SAFE", "INJECTION"]
    assert result.categories[0].score == pytest.approx(0.73)
    assert result.categories[1].score == pytest.approx(0.27)
    # Every class carries a concrete verdict: SAFE was predicted, INJECTION wasn't.
    assert result.categories[0].triggered is True
    assert result.categories[1].triggered is False


def test_match_injection_label_categories_complement_fallback() -> None:
    """Without `labels` (encoderfile), 2-class names are reconstructed via complement."""
    # Safe prediction: the complement index must be the injection class.
    safe_result = match_injection_label(
        _uniform_output(["SAFE"], np.array([[0.9, 0.1]])),
        injection_label="INJECTION",
    )
    assert [category.name for category in safe_result.categories] == ["SAFE", "INJECTION"]
    assert safe_result.score == pytest.approx(0.1)

    # Injection prediction: the complement label is unknown and gets a placeholder name.
    unsafe_result = match_injection_label(
        _uniform_output(["INJECTION"], np.array([[0.2, 0.8]])),
        injection_label="INJECTION",
    )
    assert [category.name for category in unsafe_result.categories] == ["LABEL_0", "INJECTION"]
    assert unsafe_result.score == pytest.approx(0.8)
    assert unsafe_result.categories[1].triggered is True
