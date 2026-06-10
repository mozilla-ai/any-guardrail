from unittest import mock

import pytest
from azure.ai.contentsafety.models import TextCategory

from any_guardrail.base import GuardrailOutput
from any_guardrail.guardrails.azure_content_safety.azure_content_safety import AzureContentSafety
from any_guardrail.types import GuardrailInferenceOutput


def _mock_model_outputs(
    severities: dict[str, int], blocklists_match: list[str] | None = None
) -> GuardrailInferenceOutput:  # type: ignore[type-arg]
    mock_model_outputs = mock.MagicMock()
    mock_model_outputs.categories_analysis = [
        mock.MagicMock(category=TextCategory.HATE, severity=severities["hate"]),
        mock.MagicMock(category=TextCategory.SELF_HARM, severity=severities["self_harm"]),
        mock.MagicMock(category=TextCategory.SEXUAL, severity=severities["sexual"]),
        mock.MagicMock(category=TextCategory.VIOLENCE, severity=severities["violence"]),
    ]
    mock_model_outputs.blocklists_match = blocklists_match
    return GuardrailInferenceOutput(data=mock_model_outputs)


def test_azure_content_safety_guardrail_post_processing() -> None:
    """Severities surface as categories; score is the max severity normalized to [0, 1]."""
    guardrail = AzureContentSafety(
        endpoint="https://fake-endpoint.cognitiveservices.azure.com/",
        api_key="fake-api-key",
        threshold=2,
        score_type="max",
        blocklist_names=None,
    )

    result = guardrail._post_processing(_mock_model_outputs({"hate": 0, "self_harm": 2, "sexual": 4, "violence": 6}))

    assert isinstance(result, GuardrailOutput)
    assert not result.valid
    assert result.score == pytest.approx(6 / 7)
    assert {category.name: category.severity for category in result.categories} == {
        "hate": 0,
        "self_harm": 2,
        "sexual": 4,
        "violence": 6,
    }
    assert [category.triggered for category in result.categories] == [False, True, True, True]
    assert result.categories[3].score == pytest.approx(6 / 7)
    assert result.extra is None


def test_azure_content_safety_guardrail_post_processing_with_blocklist() -> None:
    """Blocklist matches land in extra and force valid=False."""
    guardrail = AzureContentSafety(
        endpoint="https://fake-endpoint.cognitiveservices.azure.com/",
        api_key="fake-api-key",
        threshold=2,
        score_type="max",
        blocklist_names=["default"],
    )

    result = guardrail._post_processing(
        _mock_model_outputs(
            {"hate": 0, "self_harm": 2, "sexual": 4, "violence": 6},
            blocklists_match=["some inappropriate content"],
        )
    )

    assert isinstance(result, GuardrailOutput)
    assert not result.valid
    assert result.score == pytest.approx(6 / 7)
    assert result.extra == {"blocklists_match": ["some inappropriate content"]}


def test_azure_content_safety_guardrail_post_processing_below_threshold() -> None:
    """Content below the severity threshold passes."""
    guardrail = AzureContentSafety(
        endpoint="https://fake-endpoint.cognitiveservices.azure.com/",
        api_key="fake-api-key",
        threshold=5,
        score_type="max",
        blocklist_names=None,
    )

    result = guardrail._post_processing(_mock_model_outputs({"hate": 0, "self_harm": 2, "sexual": 4, "violence": 4}))

    assert isinstance(result, GuardrailOutput)
    assert result.valid
    assert result.score == pytest.approx(4 / 7)
    assert [category.triggered for category in result.categories] == [False, False, False, False]


def test_azure_content_safety_guardrail_post_processing_average_score() -> None:
    """score_type='avg' aggregates the mean severity before normalizing."""
    guardrail = AzureContentSafety(
        endpoint="https://fake-endpoint.cognitiveservices.azure.com/",
        api_key="fake-api-key",
        threshold=3,
        score_type="avg",
        blocklist_names=None,
    )

    result = guardrail._post_processing(_mock_model_outputs({"hate": 0, "self_harm": 2, "sexual": 4, "violence": 6}))

    assert isinstance(result, GuardrailOutput)
    assert not result.valid
    assert result.score == pytest.approx(3 / 7)
    # Per-category triggered still compares each raw severity to the threshold.
    assert [category.triggered for category in result.categories] == [False, False, True, True]
