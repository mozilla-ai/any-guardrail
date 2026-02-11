from unittest import mock

from azure.ai.contentsafety.models import TextCategory

from any_guardrail.base import GuardrailOutput
from any_guardrail.guardrails.azure_content_safety.azure_content_safety import AzureContentSafety
from any_guardrail.types import GuardrailInferenceOutput


def test_azure_content_safety_guardrail_post_processing() -> None:
    """Test the _post_processing method of AzureContentSafety guardrail."""
    guardrail = AzureContentSafety(
        endpoint="https://fake-endpoint.cognitiveservices.azure.com/",
        api_key="fake-api-key",
        threshold=2,
        score_type="max",
        blocklist_names=None,
    )

    # Mock model outputs
    mock_model_outputs = mock.MagicMock()
    mock_model_outputs.categories_analysis = [
        mock.MagicMock(category=TextCategory.HATE, severity=0),
        mock.MagicMock(category=TextCategory.SELF_HARM, severity=2),
        mock.MagicMock(category=TextCategory.SEXUAL, severity=4),
        mock.MagicMock(category=TextCategory.VIOLENCE, severity=6),
    ]
    mock_model_outputs.blocklists_match = None

    result = guardrail._post_processing(GuardrailInferenceOutput(data=mock_model_outputs))

    assert isinstance(result, GuardrailOutput)
    assert not result.valid
    assert result.score == 6
    assert result.explanation == {
        "hate": 0,
        "self_harm": 2,
        "sexual": 4,
        "violence": 6,
        "blocklist": None,
    }


def test_azure_content_safety_guardrail_post_processing_with_blocklist() -> None:
    """Test the _post_processing method of AzureContentSafety guardrail with blocklist match."""
    guardrail = AzureContentSafety(
        endpoint="https://fake-endpoint.cognitiveservices.azure.com/",
        api_key="fake-api-key",
        threshold=2,
        score_type="max",
        blocklist_names=["default"],
    )

    # Mock model outputs
    mock_model_outputs = mock.MagicMock()
    mock_model_outputs.categories_analysis = [
        mock.MagicMock(category=TextCategory.HATE, severity=0),
        mock.MagicMock(category=TextCategory.SELF_HARM, severity=2),
        mock.MagicMock(category=TextCategory.SEXUAL, severity=4),
        mock.MagicMock(category=TextCategory.VIOLENCE, severity=6),
    ]
    mock_model_outputs.blocklists_match = ["some inappropriate content"]

    result = guardrail._post_processing(GuardrailInferenceOutput(data=mock_model_outputs))

    assert isinstance(result, GuardrailOutput)
    assert not result.valid
    assert result.score == 6
    assert result.explanation == {
        "hate": 0,
        "self_harm": 2,
        "sexual": 4,
        "violence": 6,
        "blocklist": ["some inappropriate content"],
    }


def test_azure_content_safety_guardrail_post_processing_below_threshold() -> None:
    """Test the _post_processing method of AzureContentSafety guardrail with score below threshold."""
    guardrail = AzureContentSafety(
        endpoint="https://fake-endpoint.cognitiveservices.azure.com/",
        api_key="fake-api-key",
        threshold=5,
        score_type="max",
        blocklist_names=None,
    )

    # Mock model outputs
    mock_model_outputs = mock.MagicMock()
    mock_model_outputs.categories_analysis = [
        mock.MagicMock(category=TextCategory.HATE, severity=0),
        mock.MagicMock(category=TextCategory.SELF_HARM, severity=2),
        mock.MagicMock(category=TextCategory.SEXUAL, severity=4),
        mock.MagicMock(category=TextCategory.VIOLENCE, severity=4),
    ]
    mock_model_outputs.blocklists_match = None

    result = guardrail._post_processing(GuardrailInferenceOutput(data=mock_model_outputs))

    assert isinstance(result, GuardrailOutput)
    assert result.valid
    assert result.score == 4
    assert result.explanation == {
        "hate": 0,
        "self_harm": 2,
        "sexual": 4,
        "violence": 4,
        "blocklist": None,
    }


def test_azure_content_safety_guardrail_post_processing_average_score() -> None:
    """Test the _post_processing method of AzureContentSafety guardrail with average score calculation."""
    guardrail = AzureContentSafety(
        endpoint="https://fake-endpoint.cognitiveservices.azure.com/",
        api_key="fake-api-key",
        threshold=3,
        score_type="avg",
        blocklist_names=None,
    )

    # Mock model outputs
    mock_model_outputs = mock.MagicMock()
    mock_model_outputs.categories_analysis = [
        mock.MagicMock(category=TextCategory.HATE, severity=0),
        mock.MagicMock(category=TextCategory.SELF_HARM, severity=2),
        mock.MagicMock(category=TextCategory.SEXUAL, severity=4),
        mock.MagicMock(category=TextCategory.VIOLENCE, severity=6),
    ]
    mock_model_outputs.blocklists_match = None

    result = guardrail._post_processing(GuardrailInferenceOutput(data=mock_model_outputs))

    assert isinstance(result, GuardrailOutput)
    assert not result.valid
    assert result.score == 3.0
    assert result.explanation == {
        "hate": 0,
        "self_harm": 2,
        "sexual": 4,
        "violence": 6,
        "blocklist": None,
    }
