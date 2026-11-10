from typing import Any
from unittest.mock import Mock, patch

import pytest
import requests

from any_guardrail import AnyGuardrail, GuardrailName, GuardrailOutput
from any_guardrail.evaluate import EvaluateArgumentError
from any_guardrail.guardrails.deepkeep.deepkeep import Deepkeep
from any_guardrail.parameter_registry import get_parameter_schema
from any_guardrail.taxonomy import BackendType


def make_guard(**kwargs: Any) -> Deepkeep:
    config: dict[str, Any] = {
        "api_key": "test-credential",
        "endpoint": "https://deepkeep.example/",
        "firewall_id": "fw-input",
    }
    return Deepkeep(**(config | kwargs))


@pytest.mark.parametrize("flagged", [False, True])
def test_wire_contract(flagged: bool) -> None:
    body = {"flagged": flagged, "request_id": "req-123"}
    with patch("any_guardrail.guardrails.deepkeep.deepkeep.requests.post") as post:
        post.return_value = Mock(status_code=200, json=Mock(return_value=body))
        guard = AnyGuardrail.create(
            GuardrailName.DEEPKEEP,
            api_key="test-credential",
            endpoint="https://deepkeep.example/",
            firewall_id="fw-input",
        )
        post.assert_not_called()
        result = AnyGuardrail.evaluate(GuardrailName.DEEPKEEP, guard, "hello")
        post.assert_called_once_with(
            "https://deepkeep.example/api/v3/openai/moderations/pre",
            headers={"X-API-Key": "test-credential"},
            json={"model": "fw-input", "input": "hello", "logs": False},
            timeout=10.0,
            allow_redirects=False,
        )
    assert isinstance(result, GuardrailOutput)
    assert result.valid is not flagged
    assert result.raw == body
    assert result.score is None
    assert result.usage is not None
    assert result.usage.latency_ms is not None
    assert result.usage.latency_ms >= 0


@pytest.mark.parametrize("action", ["block", "alert", "redact", "modify", "unknown"])
def test_flagged_actions_never_forward_unchanged_text(action: str) -> None:
    body = {"flagged": True, "verbosity": [{"details": {"guardrail_action": action}}]}
    with patch("any_guardrail.guardrails.deepkeep.deepkeep.requests.post") as post:
        post.return_value = Mock(status_code=200, json=Mock(return_value=body))
        assert make_guard().validate("unsafe").valid is False


@pytest.mark.parametrize("body", [{}, None, [], {"flagged": None}, {"flagged": "false"}, {"flagged": 0}])
def test_missing_or_non_boolean_verdict_is_unavailable(body: object) -> None:
    with patch("any_guardrail.guardrails.deepkeep.deepkeep.requests.post") as post:
        post.return_value = Mock(status_code=200, json=Mock(return_value=body))
        with pytest.raises(ValueError, match="boolean flagged"):
            make_guard().validate("hello")


@pytest.mark.parametrize("status", [301, 302, 400, 401, 403, 429, 500, 503])
def test_http_errors_and_redirects_are_unavailable(status: int) -> None:
    with patch("any_guardrail.guardrails.deepkeep.deepkeep.requests.post") as post:
        post.return_value = Mock(status_code=status, text="private-content test-credential")
        with pytest.raises(ValueError, match=f"HTTP {status}") as error:
            make_guard().validate("hello")
        assert "private-content" not in str(error.value)
        assert "test-credential" not in str(error.value)


@pytest.mark.parametrize("error_type", [requests.Timeout, requests.ConnectionError])
def test_network_errors_are_sanitized(error_type: type[requests.RequestException]) -> None:
    with patch("any_guardrail.guardrails.deepkeep.deepkeep.requests.post") as post:
        post.side_effect = error_type("private-content test-credential")
        with pytest.raises(ValueError, match="request failed") as error:
            make_guard().validate("hello")
        assert "test-credential" not in str(error.value)


def test_invalid_json() -> None:
    with patch("any_guardrail.guardrails.deepkeep.deepkeep.requests.post") as post:
        post.return_value = Mock(status_code=200, json=Mock(side_effect=ValueError("private-content")))
        with pytest.raises(ValueError, match="invalid JSON"):
            make_guard().validate("hello")


@pytest.mark.parametrize(
    "config",
    [
        {"api_key": ""},
        {"firewall_id": " "},
        {"endpoint": ""},
        {"endpoint": "ftp://host"},
        {"endpoint": "https://user:credential@host"},
        {"endpoint": "https://host?key=credential"},
        {"endpoint": "https://host/#fragment"},
        {"endpoint": "https://host:invalid"},
        {"timeout": 0},
        {"timeout": -1},
        {"timeout": float("inf")},
        {"timeout": float("nan")},
        {"timeout": True},
        {"logs": "false"},
    ],
)
def test_invalid_configuration(config: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="DeepKeep"):
        make_guard(**config)


@pytest.mark.parametrize("text", [None, [], {}, 42])
def test_only_text_is_supported(text: Any) -> None:
    with patch("any_guardrail.guardrails.deepkeep.deepkeep.requests.post") as post:
        with pytest.raises(TypeError):
            make_guard().validate(text)
        post.assert_not_called()


def test_custom_timeout_logging_and_private_endpoint() -> None:
    with patch("any_guardrail.guardrails.deepkeep.deepkeep.requests.post") as post:
        post.return_value = Mock(status_code=200, json=Mock(return_value={"flagged": False}))
        make_guard(endpoint="http://deepkeep:8080", timeout=3.0, logs=True).validate("")
        assert post.call_args.args[0] == "http://deepkeep:8080/api/v3/openai/moderations/pre"
        assert post.call_args.kwargs["timeout"] == 3.0
        assert post.call_args.kwargs["json"]["logs"] is True


def test_native_catalog_metadata_and_secret_schema() -> None:
    metadata = AnyGuardrail.metadata(GuardrailName.DEEPKEEP)
    assert metadata.backend is BackendType.HOSTED_API
    assert metadata.display_name == "DeepKeep"
    assert Deepkeep.__doc__ is not None
    assert metadata.description == Deepkeep.__doc__.splitlines()[0]
    schema = {parameter.name: parameter for parameter in get_parameter_schema(GuardrailName.DEEPKEEP)}
    assert schema["api_key"].secret
    assert schema["api_key"].required
    assert schema["endpoint"].required
    assert schema["firewall_id"].required


def test_evaluate_rejects_output_screening() -> None:
    with patch("any_guardrail.guardrails.deepkeep.deepkeep.requests.post") as post:
        with pytest.raises(EvaluateArgumentError, match="no response/second-text"):
            AnyGuardrail.evaluate(GuardrailName.DEEPKEEP, make_guard(), "prompt", response="output")
        post.assert_not_called()
