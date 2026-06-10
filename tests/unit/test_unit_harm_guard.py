from unittest.mock import MagicMock

import numpy as np
import pytest

from any_guardrail.guardrails.harm_guard.harm_guard import HARMGUARD_DEFAULT_THRESHOLD, HarmGuard
from any_guardrail.types import AnyDict, GuardrailInferenceOutput


@pytest.fixture
def harm_guard_instance() -> HarmGuard:
    """Build a HarmGuard instance with a mocked provider and default threshold."""
    instance = object.__new__(HarmGuard)
    instance.provider = MagicMock()
    instance.threshold = HARMGUARD_DEFAULT_THRESHOLD
    return instance


def _inference_output(safe: float, unsafe: float) -> GuardrailInferenceOutput[AnyDict]:
    return GuardrailInferenceOutput(data={"scores": np.array([[safe, unsafe]])})


def test_unsafe_text_is_flagged(harm_guard_instance: HarmGuard) -> None:
    result = harm_guard_instance._post_processing(_inference_output(safe=0.2, unsafe=0.8))

    assert result.valid is False
    assert result.score == pytest.approx(0.8)
    assert [category.name for category in result.categories] == ["safe", "unsafe"]
    assert result.categories[0].score == pytest.approx(0.2)
    assert result.categories[1].score == pytest.approx(0.8)
    assert result.categories[1].triggered is True


def test_safe_text_passes(harm_guard_instance: HarmGuard) -> None:
    result = harm_guard_instance._post_processing(_inference_output(safe=0.95, unsafe=0.05))

    assert result.valid is True
    assert result.score == pytest.approx(0.05)
    assert result.categories[1].triggered is False


def test_threshold_is_respected(harm_guard_instance: HarmGuard) -> None:
    harm_guard_instance.threshold = 0.9

    result = harm_guard_instance._post_processing(_inference_output(safe=0.2, unsafe=0.8))

    assert result.valid is True
    assert result.categories[1].triggered is False
