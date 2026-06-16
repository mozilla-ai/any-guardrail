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
    instance.rubric = {score: f"level {score}" for score in range(6)}  # 0-5 scale
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


def test_flowjudge_normalizes_rubric_into_risk_score(flowjudge_instance: Flowjudge) -> None:
    """The 0-5 rubric is normalized onto canonical risk: higher_is_better → 5 means lowest risk."""
    # higher_is_better=True (default fixture): raw 5 → quality 1.0 → risk 0.0.
    best = flowjudge_instance._post_processing(_eval_output(5))
    assert best.score == pytest.approx(0.0)
    assert best.extra is not None
    assert best.extra["rubric_score"] == 5

    # raw 0 → quality 0.0 → risk 1.0.
    worst = flowjudge_instance._post_processing(_eval_output(0))
    assert worst.score == pytest.approx(1.0)

    # raw 4 on a 0-5 scale → quality 0.8 → risk 0.2.
    mid = flowjudge_instance._post_processing(_eval_output(4))
    assert mid.score == pytest.approx(0.2)


def test_flowjudge_lower_is_better_inverts_direction(flowjudge_instance: Flowjudge) -> None:
    flowjudge_instance.higher_is_better = False
    # raw 5 with lower-is-better → quality 1.0 → risk = quality = 1.0.
    result = flowjudge_instance._post_processing(_eval_output(5))
    assert result.score == pytest.approx(1.0)
