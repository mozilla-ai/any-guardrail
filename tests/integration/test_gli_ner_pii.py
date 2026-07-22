"""Integration test for the GliNerPii span-emitting PII/NER guardrail (issue #212).

Skipped cleanly when the optional ``gliner`` extra (``gliner2``) is not installed.
Downloads a real ~0.3B GLiNER2 model. Auto-marked ``e2e`` by the directory conftest.
"""

import pytest

from any_guardrail import AnyGuardrail, GuardrailName
from any_guardrail.types import GuardrailOutput

pytest.importorskip("gliner2", reason="requires `any-guardrail[gliner]`")


def test_gli_ner_pii_detects_and_redacts_email() -> None:
    guardrail = AnyGuardrail.create(GuardrailName.GLI_NER_PII)
    result = guardrail.validate("Contact me at john.smith@example.com for details.")
    assert isinstance(result, GuardrailOutput)
    assert result.valid is False
    assert result.spans, "expected at least one PII span"
    span = result.spans[0]
    assert span.start is not None
    assert span.end is not None
    # The flagged text lines up with its character offsets.
    original = "Contact me at john.smith@example.com for details."
    assert original[span.start : span.end] == (span.text or original[span.start : span.end])
    # A redacted copy is produced and no longer contains the raw email.
    assert result.modified_text is not None
    assert "john.smith@example.com" not in result.modified_text


def test_gli_ner_pii_benign_text_is_valid() -> None:
    guardrail = AnyGuardrail.create(GuardrailName.GLI_NER_PII)
    result = guardrail.validate("The weather is pleasant this afternoon.")
    assert isinstance(result, GuardrailOutput)
    assert result.valid is True
    assert result.spans is None
    assert result.modified_text is None
