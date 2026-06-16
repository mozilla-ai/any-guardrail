"""Tests for the token-classification / span infrastructure (utils + provider)."""

from unittest.mock import MagicMock

import pytest

from any_guardrail.guardrails.utils import spans_from_token_labels


def test_spans_bio_scheme_merges_contiguous_tokens() -> None:
    text = "Call Alice Smith now"
    # offsets for: "Call"(0,4) "Alice"(5,10) "Smith"(11,16) "now"(17,20)
    offsets = [(0, 4), (5, 10), (11, 16), (17, 20)]
    id2label = {0: "O", 1: "B-PER", 2: "I-PER"}
    label_ids = [0, 1, 2, 0]
    scores = [0.99, 0.95, 0.90, 0.99]

    spans = spans_from_token_labels(label_ids, offsets, id2label, text, scores)

    assert len(spans) == 1
    span = spans[0]
    assert (span.start, span.end) == (5, 16)
    assert span.text == "Alice Smith"
    assert span.label == "PER"
    assert span.score == pytest.approx((0.95 + 0.90) / 2)


def test_spans_bioes_single_and_separate_entities() -> None:
    text = "Bob met Carol"
    offsets = [(0, 3), (4, 7), (8, 13)]
    id2label = {0: "O", 1: "S-PER"}
    label_ids = [1, 0, 1]  # two separate single-token PER entities

    spans = spans_from_token_labels(label_ids, offsets, id2label, text)

    assert [(s.start, s.end, s.text) for s in spans] == [(0, 3, "Bob"), (8, 13, "Carol")]
    assert all(s.label == "PER" for s in spans)


def test_spans_skips_special_tokens_and_outside() -> None:
    text = "hello world"
    # First/last tokens are specials (offset (0, 0)); middle is outside.
    offsets = [(0, 0), (0, 5), (6, 11), (0, 0)]
    id2label = {0: "O", 1: "B-PII"}
    label_ids = [0, 0, 0, 0]

    assert spans_from_token_labels(label_ids, offsets, id2label, text) == []


def test_spans_two_adjacent_different_entities_do_not_merge() -> None:
    text = "email a@b.co"
    offsets = [(0, 5), (6, 12)]
    id2label = {0: "O", 1: "B-TYPE", 2: "B-EMAIL"}
    label_ids = [1, 2]

    spans = spans_from_token_labels(label_ids, offsets, id2label, text)

    assert [s.label for s in spans] == ["TYPE", "EMAIL"]


def test_provider_token_classification_output_shape() -> None:
    """``_token_classification_output`` surfaces per-token ids, scores, offsets, and id2label."""
    import numpy as np
    import torch

    from any_guardrail.providers.huggingface import HuggingFaceProvider

    provider = object.__new__(HuggingFaceProvider)
    provider.model = MagicMock()
    provider.model.config.id2label = {0: "O", 1: "B-PER"}
    # logits shape (batch=1, seq=2, num_labels=2); token 1 favors B-PER.
    logits = torch.tensor([[[2.0, 0.1], [0.1, 3.0]]])
    offsets = torch.tensor([[[0, 4], [5, 10]]])

    out = provider._token_classification_output(logits, offsets).data

    assert out["token_label_ids"] == [[0, 1]]
    assert out["offsets"] == [[[0, 4], [5, 10]]]
    assert out["id2label"] == {0: "O", 1: "B-PER"}
    assert out["predicted_indices"] is None
    assert np.asarray(out["scores"]).shape == (1, 2, 2)
