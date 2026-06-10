import numpy as np
import pytest
from pydantic import ValidationError

from any_guardrail import CategoryResult, GuardrailOutput, GuardrailUsage, SpanResult


def test_guardrail_output_new_field_defaults() -> None:
    output: GuardrailOutput = GuardrailOutput(valid=True)

    assert output.categories == []
    assert output.spans is None
    assert output.modified_text is None
    assert output.usage is None
    assert output.extra is None
    assert output.raw is None


def test_guardrail_output_holds_structured_results() -> None:
    output: GuardrailOutput = GuardrailOutput(
        valid=False,
        score=0.91,
        categories=[CategoryResult(name="S1", description="Violent Crimes", triggered=True, score=0.91)],
        spans=[SpanResult(start=0, end=4, text="text", label="PII", score=0.5)],
        usage=GuardrailUsage(model_id="some/model", latency_ms=12.5),
        extra={"rubric_score": 4},
    )

    assert output.categories[0].name == "S1"
    assert output.categories[0].triggered is True
    assert output.spans is not None
    assert output.spans[0].end == 4
    assert output.usage is not None
    assert output.usage.model_id == "some/model"
    assert output.extra == {"rubric_score": 4}


def test_guardrail_output_raw_accepts_arbitrary_objects() -> None:
    payload = np.array([0.1, 0.9])

    output: GuardrailOutput = GuardrailOutput(valid=True, raw=payload)

    assert output.raw is payload


def test_valid_required() -> None:
    with pytest.raises(ValidationError):
        GuardrailOutput()  # type: ignore[call-arg]


def test_category_result_requires_name() -> None:
    with pytest.raises(ValidationError):
        CategoryResult()  # type: ignore[call-arg]


def test_span_result_requires_offsets() -> None:
    with pytest.raises(ValidationError):
        SpanResult(start=3)  # type: ignore[call-arg]
