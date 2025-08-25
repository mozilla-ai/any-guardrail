from unittest.mock import patch

import pytest

from any_guardrail import GuardrailOutput
from any_guardrail.guardrails.glider import Glider


@pytest.mark.parametrize(
    ("model_outputs", "expected_score"),
    [
        ("<score>\n0.9\n</score>", None),
        ("<score>\nnot_a_number\n</score>", None),
        ("bad_format", None),
        ("<score>\n8\n</score>", 8),
    ],
)
def test_glider_postprocessing(model_outputs: str, expected_score: float | None) -> None:
    with patch("any_guardrail.guardrails.glider.glider.Glider._load_model"):
        glider = Glider("foo", "bar")

        result = glider._post_processing(model_outputs)

        assert isinstance(result, GuardrailOutput)
        assert result.score == expected_score
