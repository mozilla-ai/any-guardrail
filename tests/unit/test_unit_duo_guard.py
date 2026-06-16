from unittest.mock import MagicMock

import numpy as np
import pytest

from any_guardrail.guardrails.duo_guard.duo_guard import (
    DUOGUARD_CATEGORIES,
    DUOGUARD_DEFAULT_THRESHOLD,
    DuoGuard,
)
from any_guardrail.types import AnyDict, GuardrailInferenceOutput


@pytest.fixture
def duo_guard_instance() -> DuoGuard:
    """Build a DuoGuard instance with a mocked provider and default threshold."""
    instance = object.__new__(DuoGuard)
    instance.provider = MagicMock()
    instance.threshold = DUOGUARD_DEFAULT_THRESHOLD
    return instance


def _inference_output(probabilities: list[float]) -> GuardrailInferenceOutput[AnyDict]:
    return GuardrailInferenceOutput(data={"scores": np.array([probabilities])})


def test_all_categories_surface_with_probabilities(duo_guard_instance: DuoGuard) -> None:
    probabilities = [0.05] * len(DUOGUARD_CATEGORIES)
    probabilities[8] = 0.92  # "Hate"

    result = duo_guard_instance._post_processing(_inference_output(probabilities))

    assert result.valid is False
    assert result.score == pytest.approx(0.92)
    assert [category.name for category in result.categories] == DUOGUARD_CATEGORIES
    assert result.categories[8].triggered is True
    assert result.categories[8].score == pytest.approx(0.92)
    assert all(category.triggered is False for i, category in enumerate(result.categories) if i != 8)


def test_safe_text_passes_with_full_distribution(duo_guard_instance: DuoGuard) -> None:
    probabilities = [0.01 * (i + 1) for i in range(len(DUOGUARD_CATEGORIES))]

    result = duo_guard_instance._post_processing(_inference_output(probabilities))

    assert result.valid is True
    assert result.score == pytest.approx(max(probabilities))
    # Every category keeps its real probability — nothing is discarded.
    assert [category.score for category in result.categories] == pytest.approx(probabilities)


def test_custom_threshold_changes_verdict(duo_guard_instance: DuoGuard) -> None:
    duo_guard_instance.threshold = 0.95
    probabilities = [0.05] * len(DUOGUARD_CATEGORIES)
    probabilities[0] = 0.9

    result = duo_guard_instance._post_processing(_inference_output(probabilities))

    assert result.valid is True
    assert result.categories[0].triggered is False
