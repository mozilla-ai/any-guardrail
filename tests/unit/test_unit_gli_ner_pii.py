"""Unit tests for the GliNerPii span-emitting PII/NER guardrail (issue #212)."""

from typing import Any
from unittest.mock import MagicMock

import pytest

from any_guardrail.guardrails.gli_ner_pii.gli_ner_pii import DEFAULT_PII_ENTITIES, GliNerPii


def _pii(result: dict[str, Any], threshold: float = 0.5) -> GliNerPii:
    instance = object.__new__(GliNerPii)
    instance.model_id = "fastino/gliner2-privacy-filter-PII-multi"
    instance.threshold = threshold
    instance.model = MagicMock()
    instance.model.extract_entities.return_value = result
    return instance


def test_gli_ner_pii_flags_spans_and_redacts() -> None:
    guard = _pii(
        {
            "entities": {
                "email": [{"text": "john.smith@acme.com", "confidence": 0.99, "start": 6, "end": 25}],
                "phone_number": [{"text": "+1 415 555 0199", "confidence": 0.9, "start": 34, "end": 49}],
            }
        }
    )
    result = guard.validate("Email john.smith@acme.com or call +1 415 555 0199.")
    assert result.valid is False
    assert result.spans is not None
    assert len(result.spans) == 2
    # spans are sorted by offset; the first is the email
    assert result.spans[0].label == "email"
    assert result.spans[0].start == 6
    assert result.spans[0].end == 25
    assert result.spans[0].score == pytest.approx(0.99)
    assert result.score == pytest.approx(0.99)
    assert {c.name for c in result.categories} == {"email", "phone_number"}
    assert result.modified_text == "Email [REDACTED_EMAIL] or call [REDACTED_PHONE_NUMBER]."


def test_gli_ner_pii_no_pii_is_valid() -> None:
    guard = _pii({"entities": {"email": [], "person": []}})
    result = guard.validate("The weather is nice today.")
    assert result.valid is True
    assert result.spans is None
    assert result.modified_text is None
    assert result.score is None
    assert result.categories == []


def test_gli_ner_pii_forwards_entity_types_and_threshold() -> None:
    guard = _pii({"entities": {}}, threshold=0.5)
    guard.validate("text", entity_types=["email"], threshold=0.7)
    args, kwargs = guard.model.extract_entities.call_args
    assert args[0] == "text"
    assert args[1] == ["email"]
    assert kwargs["threshold"] == 0.7
    assert kwargs["include_spans"] is True
    assert kwargs["include_confidence"] is True


def test_gli_ner_pii_defaults_to_default_entities_and_instance_threshold() -> None:
    guard = _pii({"entities": {}}, threshold=0.42)
    guard.validate("text")
    args, kwargs = guard.model.extract_entities.call_args
    assert args[1] == DEFAULT_PII_ENTITIES
    assert kwargs["threshold"] == 0.42


def test_gli_ner_pii_overlapping_spans_redact_once_keeping_higher_score() -> None:
    # "person" and "name" overlap the same region; redaction must not double-replace.
    guard = _pii(
        {
            "entities": {
                "name": [{"text": "Ada Lovelace", "confidence": 0.7, "start": 0, "end": 12}],
                "person": [{"text": "Ada Lovelace", "confidence": 0.95, "start": 0, "end": 12}],
            }
        }
    )
    result = guard.validate("Ada Lovelace wrote the first program.")
    assert result.spans is not None
    assert len(result.spans) == 2  # both detections are surfaced
    # higher-scoring label (person) wins the redaction of the overlapping region
    assert result.modified_text == "[REDACTED_PERSON] wrote the first program."


def test_gli_ner_pii_custom_placeholder() -> None:
    guard = _pii({"entities": {"email": [{"text": "a@b.com", "confidence": 0.8, "start": 0, "end": 7}]}})
    result = guard.validate("a@b.com", redaction_placeholder="<PII>")
    assert result.modified_text == "<PII>"


def test_gli_ner_pii_missing_package_raises() -> None:
    import any_guardrail.guardrails.gli_ner_pii.gli_ner_pii as mod

    original = mod.MISSING_PACKAGES_ERROR
    mod.MISSING_PACKAGES_ERROR = ImportError("no gliner2")
    try:
        with pytest.raises(ImportError, match="gliner"):
            GliNerPii()
    finally:
        mod.MISSING_PACKAGES_ERROR = original
