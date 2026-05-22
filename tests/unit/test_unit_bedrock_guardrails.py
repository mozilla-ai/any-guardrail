from typing import Any
from unittest import mock

import pytest

from any_guardrail.base import GuardrailOutput
from any_guardrail.guardrails.bedrock_guardrails.bedrock_guardrails import BedrockGuardrails
from any_guardrail.types import GuardrailInferenceOutput, GuardrailPreprocessOutput


def _make_guardrail(source: str = "INPUT") -> BedrockGuardrails:
    """Instantiate BedrockGuardrails with a mocked boto3 client."""
    with mock.patch("boto3.client") as mock_boto3_client:
        mock_boto3_client.return_value = mock.MagicMock()
        return BedrockGuardrails(
            guardrail_identifier="fake-guardrail-id",
            guardrail_version="DRAFT",
            source=source,
            region_name="us-east-1",
            aws_access_key_id="fake-key",
            aws_secret_access_key="fake-secret",
        )


def test_bedrock_guardrails_invalid_source_raises() -> None:
    """Constructor must reject any source other than INPUT/OUTPUT."""
    with pytest.raises(ValueError, match="source must be one of"):
        BedrockGuardrails(guardrail_identifier="fake-id", source="BOTH")


def test_bedrock_guardrails_post_processing_action_none() -> None:
    """When the Bedrock action is NONE, the content is valid and score is 0.0."""
    guardrail = _make_guardrail()

    response: dict[str, Any] = {
        "action": "NONE",
        "outputs": [],
        "assessments": [
            {
                "topicPolicy": {"topics": []},
                "contentPolicy": {"filters": []},
                "wordPolicy": {"customWords": [], "managedWordLists": []},
                "sensitiveInformationPolicy": {"piiEntities": [], "regexes": []},
                "contextualGroundingPolicy": {"filters": []},
            }
        ],
        "usage": {"topicPolicyUnits": 1, "contentPolicyUnits": 1},
    }
    result = guardrail._post_processing(GuardrailInferenceOutput(data=response))

    assert isinstance(result, GuardrailOutput)
    assert result.valid is True
    assert result.score == 0.0
    assert result.explanation is not None
    assert result.explanation["action"] == "NONE"
    assert result.explanation["assessments"] == response["assessments"]
    assert result.explanation["outputs"] == []


def test_bedrock_guardrails_post_processing_intervened() -> None:
    """When the Bedrock action is GUARDRAIL_INTERVENED, the content is invalid and assessments are surfaced."""
    guardrail = _make_guardrail()

    assessments = [
        {
            "contentPolicy": {
                "filters": [
                    {
                        "type": "VIOLENCE",
                        "confidence": "HIGH",
                        "action": "BLOCKED",
                    },
                ]
            },
            "sensitiveInformationPolicy": {
                "piiEntities": [
                    {"type": "EMAIL", "action": "ANONYMIZED", "match": "x@example.com"}
                ],
                "regexes": [],
            },
        }
    ]
    response: dict[str, Any] = {
        "action": "GUARDRAIL_INTERVENED",
        "outputs": [{"text": "[redacted by guardrail]"}],
        "assessments": assessments,
        "usage": {"contentPolicyUnits": 1, "sensitiveInformationPolicyUnits": 1},
    }
    result = guardrail._post_processing(GuardrailInferenceOutput(data=response))

    assert isinstance(result, GuardrailOutput)
    assert result.valid is False
    assert result.score == 1.0
    assert result.explanation is not None
    assert result.explanation["action"] == "GUARDRAIL_INTERVENED"
    assert result.explanation["assessments"] == assessments
    assert result.explanation["outputs"] == [{"text": "[redacted by guardrail]"}]


def test_bedrock_guardrails_pre_processing_wraps_content() -> None:
    """_pre_processing should wrap the input text in the ApplyGuardrail content shape."""
    guardrail = _make_guardrail()
    pre = guardrail._pre_processing("hello world")
    assert isinstance(pre, GuardrailPreprocessOutput)
    assert pre.data == {"content": [{"text": {"text": "hello world"}}]}


def test_bedrock_guardrails_inference_calls_apply_guardrail() -> None:
    """_inference should call client.apply_guardrail with the configured guardrail id/version/source."""
    guardrail = _make_guardrail(source="OUTPUT")
    expected_response = {"action": "NONE", "outputs": [], "assessments": []}
    guardrail.client.apply_guardrail = mock.MagicMock(return_value=expected_response)

    pre = guardrail._pre_processing("hello world")
    result = guardrail._inference(pre)

    guardrail.client.apply_guardrail.assert_called_once_with(
        guardrailIdentifier="fake-guardrail-id",
        guardrailVersion="DRAFT",
        source="OUTPUT",
        content=[{"text": {"text": "hello world"}}],
    )
    assert result.data == expected_response


def test_bedrock_guardrails_validate_end_to_end() -> None:
    """The default validate() pipeline should chain pre/inference/post correctly."""
    guardrail = _make_guardrail()
    response = {
        "action": "GUARDRAIL_INTERVENED",
        "outputs": [{"text": "Blocked."}],
        "assessments": [{"contentPolicy": {"filters": [{"type": "HATE"}]}}],
    }
    guardrail.client.apply_guardrail = mock.MagicMock(return_value=response)

    result = guardrail.validate("some risky text")

    assert isinstance(result, GuardrailOutput)
    assert result.valid is False
    assert result.score == 1.0
    assert result.explanation is not None
    assert result.explanation["assessments"] == response["assessments"]
