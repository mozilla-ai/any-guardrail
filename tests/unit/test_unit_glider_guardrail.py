import pytest

from any_guardrail import GuardrailOutput
from any_guardrail.guardrails.glider import Glider
from any_guardrail.types import GuardrailInferenceOutput


@pytest.fixture
def glider_instance() -> Glider:
    """Create a Glider instance without loading model weights."""
    return object.__new__(Glider)


@pytest.mark.parametrize(
    ("model_outputs", "expected_score"),
    [
        ("<score>\n0.9\n</score>", None),
        ("<score>\nnot_a_number\n</score>", None),
        ("bad_format", None),
        ("<score>\n8\n</score>", 8),
    ],
)
def test_glider_postprocessing(glider_instance: Glider, model_outputs: str, expected_score: float | None) -> None:
    result = glider_instance._post_processing(GuardrailInferenceOutput(data=model_outputs))

    assert isinstance(result, GuardrailOutput)
    assert result.score == expected_score
