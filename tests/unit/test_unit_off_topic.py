import pytest
import torch

from any_guardrail.guardrails.off_topic._postprocess import off_topic_output


def test_off_topic_pair_is_flagged() -> None:
    # Logits favoring index 1 (off-topic).
    result = off_topic_output(torch.tensor([[0.0, 2.0]]))

    assert result.valid is False
    assert result.score is not None
    assert result.score > 0.5
    assert [category.name for category in result.categories] == ["on-topic", "off-topic"]
    assert result.categories[1].triggered is True


def test_on_topic_pair_passes() -> None:
    result = off_topic_output(torch.tensor([[3.0, 0.0]]))

    assert result.valid is True
    # score is canonically P(off-topic): low for an on-topic pair.
    assert result.score is not None
    assert result.score < 0.5
    assert result.categories[1].triggered is False


def test_probabilities_sum_to_one() -> None:
    result = off_topic_output(torch.tensor([[1.0, 1.0]]))

    scores = [category.score for category in result.categories]
    assert all(score is not None for score in scores)
    assert sum(score for score in scores if score is not None) == pytest.approx(1.0)
    assert result.score == result.categories[1].score
