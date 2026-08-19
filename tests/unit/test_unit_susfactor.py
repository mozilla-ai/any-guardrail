from typing import Any

import numpy as np
import pytest

from any_guardrail.guardrails.susfactor.susfactor import MAX_CONTENT_TOKENS, Susfactor, _chunk_token_ids
from any_guardrail.types import GuardrailInferenceOutput, GuardrailPreprocessOutput


def _susfactor_instance(threshold: float = 0.5) -> Susfactor:
    instance = object.__new__(Susfactor)
    instance.model_id = "0dinai/susfactor-e5-large"
    instance.threshold = threshold
    return instance


def _inference_output(chunk_scores: list[float]) -> GuardrailInferenceOutput[dict[str, list[float]]]:
    return GuardrailInferenceOutput(data={"chunk_scores": chunk_scores})


class _FakeTokenizer:
    """Minimal stand-in for a HuggingFace tokenizer, returning fixed token ids."""

    cls_token_id = 101
    sep_token_id = 102

    def __call__(self, _text: str, *, add_special_tokens: bool = False, truncation: bool = False) -> dict[str, Any]:
        return {"input_ids": [1, 2, 3]}


class _FakeIO:
    """Minimal stand-in for onnxruntime's NodeArg (only ``.name`` is used)."""

    def __init__(self, name: str) -> None:
        self.name = name


class _FakeSession:
    """Minimal stand-in for onnxruntime.InferenceSession."""

    def __init__(
        self,
        input_names: list[str],
        output_names: list[str] | None = None,
        run_return: list[np.ndarray] | None = None,
    ) -> None:
        self._inputs = [_FakeIO(name) for name in input_names]
        self._outputs = [_FakeIO(name) for name in (output_names or ["logits"])]
        self._run_return = run_return if run_return is not None else [np.array([[0.0, 0.0]])]
        self.run_calls: list[dict[str, Any]] = []

    def get_inputs(self) -> list[_FakeIO]:
        return self._inputs

    def get_outputs(self) -> list[_FakeIO]:
        return self._outputs

    def run(self, _output_names: list[str] | None, onnx_inputs: dict[str, Any]) -> list[np.ndarray]:
        self.run_calls.append(onnx_inputs)
        return self._run_return


def _softmax(logits: list[float]) -> list[float]:
    exp = np.exp(np.array(logits) - np.max(logits))
    return list(exp / exp.sum())


# --- __init__ (dependency-injected session/tokenizer) ------------------------------


def test_init_with_injected_session_and_tokenizer_skips_loading() -> None:
    session = _FakeSession(["input_ids", "attention_mask"])
    tokenizer = _FakeTokenizer()

    guardrail = Susfactor(session=session, tokenizer=tokenizer)

    assert guardrail._session is session
    assert guardrail._tokenizer is tokenizer
    assert guardrail.threshold == 0.5
    assert guardrail.model_id == "0dinai/susfactor-e5-large-onnx"


def test_init_accepts_custom_threshold() -> None:
    guardrail = Susfactor(
        threshold=0.75, session=_FakeSession(["input_ids", "attention_mask"]), tokenizer=_FakeTokenizer()
    )

    assert guardrail.threshold == 0.75


# --- _pre_processing ---------------------------------------------------------------


def test_pre_processing_builds_single_numpy_chunk_with_special_tokens() -> None:
    guardrail = Susfactor(session=_FakeSession(["input_ids", "attention_mask"]), tokenizer=_FakeTokenizer())

    result = guardrail._pre_processing("some text")

    assert isinstance(result, GuardrailPreprocessOutput)
    chunks = result.data["chunks"]
    assert len(chunks) == 1
    expected_ids = np.array([[101, 1, 2, 3, 102]], dtype=np.int64)
    np.testing.assert_array_equal(chunks[0]["input_ids"], expected_ids)
    assert chunks[0]["input_ids"].dtype == np.int64
    np.testing.assert_array_equal(chunks[0]["attention_mask"], np.ones_like(expected_ids))
    assert chunks[0]["attention_mask"].dtype == np.int64


# --- _inference ----------------------------------------------------------------------


def test_inference_computes_softmax_score_without_token_type_ids() -> None:
    session = _FakeSession(["input_ids", "attention_mask"], run_return=[np.array([[2.0, 5.0]])])
    guardrail = Susfactor(session=session, tokenizer=_FakeTokenizer())
    input_ids = np.array([[101, 1, 2, 3, 102]], dtype=np.int64)
    model_inputs = GuardrailPreprocessOutput(
        data={"chunks": [{"input_ids": input_ids, "attention_mask": np.ones_like(input_ids)}]}
    )

    result = guardrail._inference(model_inputs)

    expected_score = _softmax([2.0, 5.0])[1]
    assert result.data["chunk_scores"] == pytest.approx([expected_score])
    assert "token_type_ids" not in session.run_calls[0]


def test_inference_adds_zero_token_type_ids_when_required_by_graph() -> None:
    session = _FakeSession(["input_ids", "attention_mask", "token_type_ids"], run_return=[np.array([[1.0, 1.0]])])
    guardrail = Susfactor(session=session, tokenizer=_FakeTokenizer())
    input_ids = np.array([[101, 1, 2, 3, 102]], dtype=np.int64)
    model_inputs = GuardrailPreprocessOutput(
        data={"chunks": [{"input_ids": input_ids, "attention_mask": np.ones_like(input_ids)}]}
    )

    guardrail._inference(model_inputs)

    onnx_inputs = session.run_calls[0]
    assert "token_type_ids" in onnx_inputs
    np.testing.assert_array_equal(onnx_inputs["token_type_ids"], np.zeros_like(input_ids))
    assert onnx_inputs["token_type_ids"].dtype == np.int64


def test_inference_resolves_logits_output_by_name_not_position() -> None:
    session = _FakeSession(
        ["input_ids", "attention_mask"],
        output_names=["other_output", "logits"],
        run_return=[np.array([[9.0, 9.0]]), np.array([[1.0, 4.0]])],
    )
    guardrail = Susfactor(session=session, tokenizer=_FakeTokenizer())
    input_ids = np.array([[101, 1, 2, 3, 102]], dtype=np.int64)
    model_inputs = GuardrailPreprocessOutput(
        data={"chunks": [{"input_ids": input_ids, "attention_mask": np.ones_like(input_ids)}]}
    )

    result = guardrail._inference(model_inputs)

    expected_score = _softmax([1.0, 4.0])[1]
    assert result.data["chunk_scores"] == pytest.approx([expected_score])


def test_inference_handles_multiple_chunks() -> None:
    session = _FakeSession(
        ["input_ids", "attention_mask"],
        run_return=[np.array([[0.0, 0.0]])],
    )
    call_returns = [np.array([[2.0, 5.0]]), np.array([[5.0, 2.0]])]

    def fake_run(_output_names: list[str] | None, onnx_inputs: dict[str, Any]) -> list[np.ndarray]:
        session.run_calls.append(onnx_inputs)
        return [call_returns[len(session.run_calls) - 1]]

    session.run = fake_run  # type: ignore[method-assign]
    guardrail = Susfactor(session=session, tokenizer=_FakeTokenizer())
    input_ids = np.array([[101, 1, 2, 3, 102]], dtype=np.int64)
    chunk = {"input_ids": input_ids, "attention_mask": np.ones_like(input_ids)}
    model_inputs = GuardrailPreprocessOutput(data={"chunks": [chunk, chunk]})

    result = guardrail._inference(model_inputs)

    expected = [_softmax([2.0, 5.0])[1], _softmax([5.0, 2.0])[1]]
    assert result.data["chunk_scores"] == pytest.approx(expected)


# --- _post_processing -----------------------------------------------------------


def test_single_chunk_benign_is_valid() -> None:
    instance = _susfactor_instance()

    result = instance._post_processing(_inference_output([0.1]))

    assert result.valid is True
    assert result.score == 0.1
    assert result.categories[0].name == "suspicious"
    assert result.categories[0].triggered is False


def test_single_chunk_suspicious_is_invalid() -> None:
    instance = _susfactor_instance()

    result = instance._post_processing(_inference_output([0.9]))

    assert result.valid is False
    assert result.score == 0.9
    assert result.categories[0].triggered is True


def test_multiple_chunks_any_exceeding_threshold_flags_the_whole_input() -> None:
    instance = _susfactor_instance()

    result = instance._post_processing(_inference_output([0.2, 0.9, 0.3]))

    assert result.valid is False
    assert result.score == 0.9
    assert result.categories[0].triggered is True


def test_multiple_chunks_all_below_threshold_is_valid() -> None:
    instance = _susfactor_instance()

    result = instance._post_processing(_inference_output([0.1, 0.2, 0.3]))

    assert result.valid is True
    assert result.score == 0.3
    assert result.categories[0].triggered is False


def test_boundary_score_exactly_at_threshold_is_flagged() -> None:
    instance = _susfactor_instance(threshold=0.5)

    result = instance._post_processing(_inference_output([0.5]))

    assert result.valid is False
    assert result.categories[0].triggered is True


def test_usage_records_model_id() -> None:
    instance = _susfactor_instance()

    result = instance._post_processing(_inference_output([0.1]))

    assert result.usage is not None
    assert result.usage.model_id == "0dinai/susfactor-e5-large"


# --- _chunk_token_ids -------------------------------------------------------------


def test_chunk_token_ids_short_input_is_single_chunk() -> None:
    token_ids = list(range(100))

    chunks = _chunk_token_ids(token_ids)

    assert chunks == [token_ids]


def test_chunk_token_ids_exactly_at_limit_is_single_chunk() -> None:
    token_ids = list(range(MAX_CONTENT_TOKENS))

    chunks = _chunk_token_ids(token_ids)

    assert chunks == [token_ids]


def test_chunk_token_ids_splits_with_overlap() -> None:
    token_ids = list(range(1000))

    chunks = _chunk_token_ids(token_ids)

    assert len(chunks) == 3
    assert chunks[0] == list(range(510))
    assert chunks[1] == list(range(460, 970))
    assert chunks[2] == list(range(920, 1000))


def test_chunk_token_ids_one_over_limit_produces_two_chunks() -> None:
    token_ids = list(range(MAX_CONTENT_TOKENS + 1))

    chunks = _chunk_token_ids(token_ids)

    assert len(chunks) == 2
    assert chunks[0] == list(range(MAX_CONTENT_TOKENS))
    assert chunks[1] == list(range(460, MAX_CONTENT_TOKENS + 1))


def test_chunk_token_ids_covers_every_token() -> None:
    token_ids = list(range(1234))

    chunks = _chunk_token_ids(token_ids)

    covered: set[int] = set()
    for chunk in chunks:
        covered.update(chunk)
    assert covered == set(token_ids)
