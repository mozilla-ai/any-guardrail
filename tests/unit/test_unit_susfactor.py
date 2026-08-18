from any_guardrail.guardrails.susfactor.susfactor import MAX_CONTENT_TOKENS, Susfactor, _chunk_token_ids
from any_guardrail.types import GuardrailInferenceOutput


def _susfactor_instance(threshold: float = 0.5) -> Susfactor:
    instance = object.__new__(Susfactor)
    instance.model_id = "0dinai/susfactor-e5-large"
    instance.threshold = threshold
    return instance


def _inference_output(chunk_scores: list[float]) -> GuardrailInferenceOutput[dict[str, list[float]]]:
    return GuardrailInferenceOutput(data={"chunk_scores": chunk_scores})


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
