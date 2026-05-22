from unittest import mock

import pytest

from any_guardrail.base import GuardrailOutput
from any_guardrail.guardrails.openai_moderation.openai_moderation import OpenaiModeration
from any_guardrail.types import GuardrailInferenceOutput


def _mock_response(flagged: bool, category_scores: dict[str, float]) -> mock.MagicMock:
    """Build a mock OpenAI moderation API response."""
    categories = {k: v >= 0.5 for k, v in category_scores.items()}

    scores_obj = mock.MagicMock()
    scores_obj.model_dump.return_value = dict(category_scores)

    categories_obj = mock.MagicMock()
    categories_obj.model_dump.return_value = dict(categories)

    result = mock.MagicMock()
    result.flagged = flagged
    result.categories = categories_obj
    result.category_scores = scores_obj

    response = mock.MagicMock()
    response.results = [result]
    return response


def test_openai_moderation_flagged_above_threshold() -> None:
    """Flagged input above default threshold returns valid=False with max score."""
    with mock.patch(
        "any_guardrail.guardrails.openai_moderation.openai_moderation.OpenAI"
    ) as mock_openai:
        guardrail = OpenaiModeration(api_key="fake-key")

        scores = {
            "hate": 0.1,
            "hate_threatening": 0.05,
            "harassment": 0.2,
            "harassment_threatening": 0.1,
            "self_harm": 0.0,
            "self_harm_intent": 0.0,
            "self_harm_instructions": 0.0,
            "sexual": 0.05,
            "sexual_minors": 0.0,
            "violence": 0.85,
            "violence_graphic": 0.4,
        }
        mock_openai.return_value.moderations.create.return_value = _mock_response(
            flagged=True, category_scores=scores
        )

        result = guardrail.validate("some violent text")

        assert isinstance(result, GuardrailOutput)
        assert result.valid is False
        assert result.score == pytest.approx(0.85)
        assert result.explanation == pytest.approx(scores)


def test_openai_moderation_safe_below_threshold() -> None:
    """Safe input below threshold returns valid=True."""
    with mock.patch(
        "any_guardrail.guardrails.openai_moderation.openai_moderation.OpenAI"
    ) as mock_openai:
        guardrail = OpenaiModeration(api_key="fake-key")

        scores = {
            "hate": 0.01,
            "harassment": 0.02,
            "self_harm": 0.0,
            "sexual": 0.05,
            "violence": 0.04,
        }
        mock_openai.return_value.moderations.create.return_value = _mock_response(
            flagged=False, category_scores=scores
        )

        result = guardrail.validate("the weather is nice today")

        assert isinstance(result, GuardrailOutput)
        assert result.valid is True
        assert result.score == pytest.approx(0.05)
        assert result.explanation == pytest.approx(scores)


def test_openai_moderation_custom_threshold_respected() -> None:
    """A custom threshold can flip the verdict even when OpenAI did not flag."""
    with mock.patch(
        "any_guardrail.guardrails.openai_moderation.openai_moderation.OpenAI"
    ) as mock_openai:
        guardrail = OpenaiModeration(api_key="fake-key", threshold=0.1)

        scores = {
            "hate": 0.15,
            "harassment": 0.05,
            "violence": 0.03,
        }
        mock_openai.return_value.moderations.create.return_value = _mock_response(
            flagged=False, category_scores=scores
        )

        result = guardrail.validate("mildly edgy text")

        assert isinstance(result, GuardrailOutput)
        assert result.valid is False
        assert result.score == pytest.approx(0.15)


def test_openai_moderation_post_processing_directly() -> None:
    """Exercise _post_processing directly to cover the scores-dict translation."""
    with mock.patch(
        "any_guardrail.guardrails.openai_moderation.openai_moderation.OpenAI"
    ):
        guardrail = OpenaiModeration(api_key="fake-key", threshold=0.5)

        scores = {"hate": 0.9, "violence": 0.2}
        response = _mock_response(flagged=True, category_scores=scores)

        result = guardrail._post_processing(GuardrailInferenceOutput(data=response))

        assert isinstance(result, GuardrailOutput)
        assert result.valid is False
        assert result.score == pytest.approx(0.9)
        assert result.explanation == pytest.approx({"hate": 0.9, "violence": 0.2})


def test_openai_moderation_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Constructor raises if neither api_key nor OPENAI_API_KEY env var is set."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OpenAI API key not provided"):
        OpenaiModeration()


def test_openai_moderation_invalid_model_id_raises() -> None:
    """An unsupported model_id should be rejected via the default() helper."""
    with mock.patch(
        "any_guardrail.guardrails.openai_moderation.openai_moderation.OpenAI"
    ):
        with pytest.raises(ValueError, match="Only supports"):
            OpenaiModeration(api_key="fake-key", model_id="not-a-real-model")
