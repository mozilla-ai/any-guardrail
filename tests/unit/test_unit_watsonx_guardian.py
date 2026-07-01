from typing import Any
from unittest import mock

import pytest

from any_guardrail.base import GuardrailOutput
from any_guardrail.guardrails.watsonx_guardian.watsonx_guardian import WatsonxGuardian

# The ``ibm-watsonx-ai`` SDK is an optional extra and is not installed in the
# unit-test environment. Constructing a real client would import it (and perform
# IAM auth), so these tests build a bare instance via ``object.__new__`` and
# exercise the pure request/response logic directly — the same approach the
# Alinia unit tests use.


def _instance(detectors: dict[str, Any] | None = None) -> WatsonxGuardian:
    guardrail = object.__new__(WatsonxGuardian)
    guardrail.model_id = "granite_guardian"
    guardrail.detectors = detectors or {"granite_guardian": {}}
    return guardrail


def _detection(detection: str, *, start: int, end: int, score: float) -> dict[str, Any]:
    return {
        "start": start,
        "end": end,
        "detection_type": "granite_guardian",
        "detection": detection,
        "score": score,
    }


def test_watsonx_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Credential validation runs before the SDK import, so a missing key is a clear ValueError."""
    monkeypatch.delenv("WATSONX_APIKEY", raising=False)
    with pytest.raises(ValueError, match="WATSONX_APIKEY"):
        WatsonxGuardian()


def test_watsonx_missing_url_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WATSONX_URL", raising=False)
    with pytest.raises(ValueError, match="WATSONX_URL"):
        WatsonxGuardian(api_key="k")


def test_watsonx_missing_project_or_space_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WATSONX_PROJECT_ID", raising=False)
    monkeypatch.delenv("WATSONX_SPACE_ID", raising=False)
    with pytest.raises(ValueError, match="project or space"):
        WatsonxGuardian(api_key="k", url="https://us-south.ml.cloud.ibm.com")


def test_watsonx_safe_text_returns_valid() -> None:
    guardrail = _instance()
    result = guardrail._post_processing({"detections": []})

    assert isinstance(result, GuardrailOutput)
    assert result.valid is True
    assert result.score == 0.0
    assert result.categories == []
    assert result.spans is None
    assert result.raw == {"detections": []}


def test_watsonx_detections_map_to_categories_and_spans() -> None:
    guardrail = _instance()
    response = {
        "detections": [
            _detection("harm", start=0, end=12, score=0.96),
            _detection("social_bias", start=20, end=30, score=0.72),
        ]
    }
    result = guardrail._post_processing(response)

    assert result.valid is False
    assert result.score == pytest.approx(0.96)
    by_name = {c.name: c for c in result.categories}
    assert by_name["harm"].triggered is True
    assert by_name["harm"].score == pytest.approx(0.96)
    assert by_name["harm"].description == "granite_guardian"
    assert result.spans is not None
    assert len(result.spans) == 2
    assert result.spans[0].start == 0
    assert result.spans[0].end == 12
    assert result.spans[0].label == "harm"


def test_watsonx_unparseable_response_fails_closed() -> None:
    guardrail = _instance()
    result = guardrail._post_processing({"unexpected": "shape"})

    assert result.valid is False
    assert result.extra == {"parse_failure": True}


def test_watsonx_null_detections_fails_closed() -> None:
    """A present-but-null ``detections`` cannot be parsed, so it is not a safe verdict."""
    guardrail = _instance()
    result = guardrail._post_processing({"detections": None})

    assert result.valid is False
    assert result.extra == {"parse_failure": True}


def test_watsonx_flagged_but_unscored_reports_none_score() -> None:
    """Detections present but with no parseable score -> flagged with score=None, not 0.0."""
    guardrail = _instance()
    result = guardrail._post_processing({"detections": [{"detection": "harm", "start": 0, "end": 4}]})

    assert result.valid is False
    assert result.score is None
    assert result.categories[0].triggered is True


def test_watsonx_validate_calls_detect_and_stamps_usage() -> None:
    guardrail = _instance()
    guardrail.guardian = mock.MagicMock()
    guardrail.guardian.detect.return_value = {"detections": [_detection("harm", start=0, end=5, score=0.9)]}

    result = guardrail.validate("some harmful text")

    guardrail.guardian.detect.assert_called_once_with(text="some harmful text")
    assert result.valid is False
    assert result.score == pytest.approx(0.9)
    assert result.usage is not None
    assert result.usage.latency_ms is not None
    assert result.usage.model_id == "granite_guardian"
