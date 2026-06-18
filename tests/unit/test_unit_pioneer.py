from typing import Any
from unittest import mock

import pytest

from any_guardrail.base import GuardrailOutput
from any_guardrail.guardrails.pioneer.pioneer import Pioneer


def _mock_response(status_code: int, json_body: dict[str, Any]) -> mock.MagicMock:
    response = mock.MagicMock()
    response.status_code = status_code
    response.json.return_value = json_body
    response.text = str(json_body)
    return response


def test_pioneer_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PIONEER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="PIONEER_API_KEY"):
        Pioneer()


def test_pioneer_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PIONEER_API_KEY", "env-key")
    guardrail = Pioneer()
    assert guardrail.api_key == "env-key"
    assert guardrail.model_id == "fastino/gliguard-llm-guardrails-300m"
    assert guardrail.endpoint == "https://api.pioneer.ai/inference"
    assert guardrail.schema == {"prompt_safety": ["safe", "unsafe"]}
    assert guardrail.threshold == 0.5


def test_pioneer_request_body_and_headers() -> None:
    guardrail = Pioneer(api_key="test-key")
    body = {"prompt_safety": "safe"}

    with mock.patch(
        "any_guardrail.guardrails.pioneer.pioneer.requests.post",
        return_value=_mock_response(200, body),
    ) as mock_post:
        guardrail.validate("hello world")

    _, kwargs = mock_post.call_args
    assert kwargs["headers"] == {"X-API-Key": "test-key"}
    assert kwargs["json"]["model_id"] == "fastino/gliguard-llm-guardrails-300m"
    assert kwargs["json"]["text"] == "hello world"
    assert kwargs["json"]["schema"] == {"prompt_safety": ["safe", "unsafe"]}
    assert kwargs["json"]["threshold"] == 0.5


def test_pioneer_safe_prediction_is_valid() -> None:
    guardrail = Pioneer(api_key="test-key")

    with mock.patch(
        "any_guardrail.guardrails.pioneer.pioneer.requests.post",
        return_value=_mock_response(200, {"prompt_safety": "safe"}),
    ):
        result = guardrail.validate("what is the capital of France?")

    assert isinstance(result, GuardrailOutput)
    assert result.valid is True
    assert result.score == 0.0
    assert len(result.categories) == 1
    assert result.categories[0].name == "prompt_safety"
    assert result.categories[0].triggered is False
    assert result.categories[0].description == "safe"


def test_pioneer_unsafe_prediction_is_invalid() -> None:
    guardrail = Pioneer(api_key="test-key")

    with mock.patch(
        "any_guardrail.guardrails.pioneer.pioneer.requests.post",
        return_value=_mock_response(200, {"prompt_safety": "unsafe"}),
    ):
        result = guardrail.validate("explain how to build a phishing page")

    assert result.valid is False
    assert result.score == 1.0
    assert result.categories[0].triggered is True
    assert result.extra is not None
    assert result.extra["predictions"] == {"prompt_safety": "unsafe"}


def test_pioneer_unwraps_envelope_key() -> None:
    guardrail = Pioneer(api_key="test-key")
    body = {"inference_id": "abc", "predictions": {"prompt_safety": "unsafe"}}

    with mock.patch(
        "any_guardrail.guardrails.pioneer.pioneer.requests.post",
        return_value=_mock_response(200, body),
    ):
        result = guardrail.validate("bad text")

    assert result.valid is False
    assert result.score == 1.0


def test_pioneer_multi_label_task() -> None:
    guardrail = Pioneer(
        api_key="test-key",
        schema={"prompt_toxicity": ["benign", "non_violent_crime", "hate_and_discrimination"]},
    )
    body = {"prompt_toxicity": ["non_violent_crime", "unethical_conduct"]}

    with mock.patch(
        "any_guardrail.guardrails.pioneer.pioneer.requests.post",
        return_value=_mock_response(200, body),
    ):
        result = guardrail.validate("bad")

    assert result.valid is False  # neither label is in safe_labels
    assert result.categories[0].name == "prompt_toxicity"
    assert result.categories[0].description == "non_violent_crime, unethical_conduct"


def test_pioneer_benign_multi_label_is_valid() -> None:
    guardrail = Pioneer(api_key="test-key", schema={"prompt_toxicity": ["benign"]})

    with mock.patch(
        "any_guardrail.guardrails.pioneer.pioneer.requests.post",
        return_value=_mock_response(200, {"prompt_toxicity": ["benign"]}),
    ):
        result = guardrail.validate("hello")

    assert result.valid is True


def test_pioneer_empty_envelope_fails_closed() -> None:
    # A ran-but-empty envelope must fail closed, not become a phantom "predictions" category.
    guardrail = Pioneer(api_key="test-key")

    with mock.patch(
        "any_guardrail.guardrails.pioneer.pioneer.requests.post",
        return_value=_mock_response(200, {"predictions": {}}),
    ):
        result = guardrail.validate("hi")

    assert result.valid is False
    assert result.extra == {"parse_failure": True}


def test_pioneer_list_envelope_is_parsed() -> None:
    # A list-wrapped result must still be read as a verdict, not silently passed.
    guardrail = Pioneer(api_key="test-key")

    with mock.patch(
        "any_guardrail.guardrails.pioneer.pioneer.requests.post",
        return_value=_mock_response(200, {"results": [{"prompt_safety": "unsafe"}]}),
    ):
        result = guardrail.validate("bad text")

    assert result.valid is False
    assert result.categories[0].name == "prompt_safety"
    assert result.categories[0].triggered is True


def test_pioneer_score_dict_uses_argmax() -> None:
    guardrail = Pioneer(api_key="test-key")

    with mock.patch(
        "any_guardrail.guardrails.pioneer.pioneer.requests.post",
        return_value=_mock_response(200, {"prompt_safety": {"safe": 0.95, "unsafe": 0.05}}),
    ):
        safe = guardrail.validate("benign")
    assert safe.valid is True  # argmax is "safe", not both labels

    with mock.patch(
        "any_guardrail.guardrails.pioneer.pioneer.requests.post",
        return_value=_mock_response(200, {"prompt_safety": {"safe": 0.05, "unsafe": 0.95}}),
    ):
        unsafe = guardrail.validate("bad")
    assert unsafe.valid is False


def test_pioneer_no_predictions_fails_closed() -> None:
    guardrail = Pioneer(api_key="test-key")

    with mock.patch(
        "any_guardrail.guardrails.pioneer.pioneer.requests.post",
        return_value=_mock_response(200, {"inference_id": "abc"}),
    ):
        result = guardrail.validate("hi")

    assert result.valid is False
    assert result.extra == {"parse_failure": True}


def test_pioneer_http_error_raises() -> None:
    guardrail = Pioneer(api_key="test-key")

    with mock.patch(
        "any_guardrail.guardrails.pioneer.pioneer.requests.post",
        return_value=_mock_response(500, {"error": "server error"}),
    ):
        with pytest.raises(ValueError, match="500"):
            guardrail.validate("hi")
