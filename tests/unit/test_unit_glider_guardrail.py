import pytest

from any_guardrail import GuardrailOutput
from any_guardrail.guardrails.glider import Glider
from any_guardrail.types import GuardrailInferenceOutput


@pytest.fixture
def glider_instance() -> Glider:
    """Create a Glider instance without loading model weights."""
    instance = object.__new__(Glider)
    instance.pass_threshold = 5
    instance.higher_is_better = True
    return instance


GENERATION = """<reasoning>
- The text is clear and well structured.
</reasoning>
<highlight>
["clear", "well structured"]
</highlight>
<score>
8
</score>"""


def test_glider_parses_reasoning_highlights_and_score(glider_instance: Glider) -> None:
    result = glider_instance._post_processing(GuardrailInferenceOutput(data=GENERATION))

    assert isinstance(result, GuardrailOutput)
    assert result.valid is True  # 8 >= 5
    assert result.explanation == "- The text is clear and well structured."
    assert result.extra is not None
    assert result.extra["rubric_score"] == 8
    assert result.extra["highlights"] == '["clear", "well structured"]'


@pytest.mark.parametrize(
    ("pass_threshold", "higher_is_better", "expected_valid"),
    [
        (5, True, True),  # 8 >= 5
        (9, True, False),  # 8 < 9
        (5, False, False),  # 8 > 5 with lower-is-better
        (8, False, True),  # 8 <= 8 with lower-is-better
    ],
)
def test_glider_threshold_mapping(
    glider_instance: Glider, pass_threshold: int, higher_is_better: bool, expected_valid: bool
) -> None:
    glider_instance.pass_threshold = pass_threshold
    glider_instance.higher_is_better = higher_is_better

    result = glider_instance._post_processing(GuardrailInferenceOutput(data=GENERATION))

    assert result.valid is expected_valid


@pytest.mark.parametrize(
    "model_outputs",
    [
        "<score>\n0.9\n</score>",
        "<score>\nnot_a_number\n</score>",
        "bad_format",
    ],
)
def test_glider_fails_closed_on_unparseable_score(glider_instance: Glider, model_outputs: str) -> None:
    result = glider_instance._post_processing(GuardrailInferenceOutput(data=model_outputs))

    assert result.valid is False
    assert result.extra == {"parse_failure": True}
    # The full generation is preserved for debugging.
    assert result.explanation == model_outputs


def test_glider_falls_back_to_full_text_without_reasoning_tags(glider_instance: Glider) -> None:
    result = glider_instance._post_processing(GuardrailInferenceOutput(data="<score>\n3\n</score>"))

    assert result.valid is False  # 3 < 5
    assert result.explanation == "<score>\n3\n</score>"
    assert result.extra is not None
    assert result.extra["rubric_score"] == 3
    assert result.extra["highlights"] is None
