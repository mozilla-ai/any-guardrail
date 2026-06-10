from unittest import mock

import pytest

from any_guardrail.guardrails.alinia.alinia import Alinia


def _post_process(response_json: dict) -> object:  # type: ignore[type-arg]
    instance = object.__new__(Alinia)
    response = mock.Mock()
    response.json.return_value = response_json
    return instance._post_processing(response)


def test_flagged_response_with_category_scores() -> None:
    result = _post_process(
        {
            "result": {
                "flagged": True,
                "category_details": {
                    "security": {"prompt_injection": 0.97, "jailbreak": 0.12},
                },
            }
        }
    )

    assert result.valid is False  # type: ignore[attr-defined]
    assert result.score == pytest.approx(0.97)  # type: ignore[attr-defined]
    assert {category.name: category.score for category in result.categories} == {  # type: ignore[attr-defined]
        "security/prompt_injection": pytest.approx(0.97),
        "security/jailbreak": pytest.approx(0.12),
    }


def test_boolean_category_details_become_triggered_flags() -> None:
    result = _post_process(
        {
            "result": {
                "flagged": False,
                "category_details": {"compliance": {"topic_violation": False}},
            }
        }
    )

    assert result.valid is True  # type: ignore[attr-defined]
    assert result.score is None  # type: ignore[attr-defined]
    assert result.categories[0].name == "compliance/topic_violation"  # type: ignore[attr-defined]
    assert result.categories[0].triggered is False  # type: ignore[attr-defined]


def test_recommendation_lands_in_explanation_and_raw_keeps_everything() -> None:
    payload = {
        "result": {"flagged": True, "category_details": {}},
        "recommendation": "Mask the detected account number before proceeding.",
    }

    result = _post_process(payload)

    assert result.explanation == "Mask the detected account number before proceeding."  # type: ignore[attr-defined]
    assert result.raw == payload  # type: ignore[attr-defined]
