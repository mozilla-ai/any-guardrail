from unittest import mock

from any_guardrail.base import GuardrailOutput

def test_azure_content_safety_guardrail_post_processing() -> None:
    """Test the _post_processing method of AzureContentSafety guardrail."""
    from any_guardrail.guardrails.azure_content_safety.azure_content_safety import AzureContentSafety

    guardrail = AzureContentSafety(
        endpoint="https://fake-endpoint.cognitiveservices.azure.com/",
        api_key="fake-api-key",
        threshold=2,
        score_type="max",
        blocklist_name=None,
    )

    # Mock model outputs
    mock_model_outputs = mock.MagicMock()
    mock_model_outputs.categories_analysis = [
        mock.MagicMock(category="Hate", severity=0),
        mock.MagicMock(category="SelfHarm", severity=2),
        mock.MagicMock(category="Sexual", severity=4),
        mock.MagicMock(category="Violence", severity=6),
    ]
    mock_model_outputs.blocklists_match = None

    result = guardrail._post_processing(mock_model_outputs)

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
    from any_guardrail.guardrails.azure_content_safety.azure_content_safety import AzureContentSafety

    guardrail = AzureContentSafety(
        endpoint="https://fake-endpoint.cognitiveservices.azure.com/",
        api_key="fake-api-key",
        threshold=2,
        score_type="max",
        blocklist_name=["default"],
    )

    # Mock model outputs
    mock_model_outputs = mock.MagicMock()
    mock_model_outputs.categories_analysis = [
        mock.MagicMock(category="Hate", severity=0),
        mock.MagicMock(category="SelfHarm", severity=2),
        mock.MagicMock(category="Sexual", severity=4),
        mock.MagicMock(category="Violence", severity=6),
    ]
    mock_model_outputs.blocklists_match = ["some inappropriate content"]

    result = guardrail._post_processing(mock_model_outputs)

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
    from any_guardrail.guardrails.azure_content_safety.azure_content_safety import AzureContentSafety

    guardrail = AzureContentSafety(
        endpoint="https://fake-endpoint.cognitiveservices.azure.com/",
        api_key="fake-api-key",
        threshold=5,
        score_type="max",
        blocklist_name=None,
    )

    # Mock model outputs
    mock_model_outputs = mock.MagicMock()
    mock_model_outputs.categories_analysis = [
        mock.MagicMock(category="Hate", severity=0),
        mock.MagicMock(category="SelfHarm", severity=2),
        mock.MagicMock(category="Sexual", severity=4),
        mock.MagicMock(category="Violence", severity=4),
    ]
    mock_model_outputs.blocklists_match = None

    result = guardrail._post_processing(mock_model_outputs)

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
    from any_guardrail.guardrails.azure_content_safety.azure_content_safety import AzureContentSafety

    guardrail = AzureContentSafety(
        endpoint="https://fake-endpoint.cognitiveservices.azure.com/",
        api_key="fake-api-key",
        threshold=3,
        score_type="avg",
        blocklist_name=None,
    )

    # Mock model outputs
    mock_model_outputs = mock.MagicMock()
    mock_model_outputs.categories_analysis = [
        mock.MagicMock(category="Hate", severity=0),
        mock.MagicMock(category="SelfHarm", severity=2),
        mock.MagicMock(category="Sexual", severity=4),
        mock.MagicMock(category="Violence", severity=6),
    ]
    mock_model_outputs.blocklists_match = None

    result = guardrail._post_processing(mock_model_outputs)

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
