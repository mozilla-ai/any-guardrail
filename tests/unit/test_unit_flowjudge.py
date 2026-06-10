from types import SimpleNamespace
from typing import Any

import pytest

from any_guardrail.guardrails.flowjudge.flowjudge import Flowjudge
from any_guardrail.types import GuardrailInferenceOutput


@pytest.fixture
def flowjudge_instance() -> Flowjudge:
    """Create a Flowjudge instance without loading the flow-judge model."""
    instance = object.__new__(Flowjudge)
    instance.pass_threshold = 3
    instance.higher_is_better = True
    return instance


def _eval_output(score: int | None, feedback: str = "Some feedback.") -> GuardrailInferenceOutput[Any]:
    """Mimic flow_judge's EvalOutput shape (.score, .feedback) without the package types."""
    return GuardrailInferenceOutput(data=SimpleNamespace(score=score, feedback=feedback))


@pytest.mark.parametrize(
    ("score", "pass_threshold", "higher_is_better", "expected_valid"),
    [
        (4, 3, True, True),
        (2, 3, True, False),
        (3, 3, True, True),
        (2, 3, False, True),
        (4, 3, False, False),
    ],
)
def test_flowjudge_threshold_mapping(
    flowjudge_instance: Flowjudge,
    score: int,
    pass_threshold: int,
    higher_is_better: bool,
    expected_valid: bool,
) -> None:
    flowjudge_instance.pass_threshold = pass_threshold
    flowjudge_instance.higher_is_better = higher_is_better

    result = flowjudge_instance._post_processing(_eval_output(score))

    assert result.valid is expected_valid
    assert result.explanation == "Some feedback."
    assert result.extra is not None
    assert result.extra["rubric_score"] == score


def test_flowjudge_fails_closed_without_score(flowjudge_instance: Flowjudge) -> None:
    result = flowjudge_instance._post_processing(_eval_output(None))

    assert result.valid is False
    assert result.extra == {"parse_failure": True}
