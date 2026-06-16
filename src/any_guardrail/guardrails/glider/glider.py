import re
from typing import ClassVar

from transformers import pipeline

from any_guardrail.base import GuardrailOutput, ThreeStageGuardrail
from any_guardrail.guardrails.utils import default, normalize_rubric_to_risk
from any_guardrail.providers.base import StandardProvider
from any_guardrail.types import ChatMessages, GuardrailInferenceOutput, GuardrailPreprocessOutput

SCORE_PATTERN = re.compile(r"<score>\s*(\d+)\s*</score>")
REASONING_PATTERN = re.compile(r"<reasoning>\s*(.*?)\s*</reasoning>", re.DOTALL)
HIGHLIGHT_PATTERN = re.compile(r"<highlight>\s*(.*?)\s*</highlight>", re.DOTALL)

SYSTEM_PROMPT_GLIDER = """
Analyze the following pass criteria carefully and score the text based on the rubric defined below.

To perform this evaluation, you must:

1. Understand the text tags, pass criteria and rubric thoroughly.
2. Review the finer details of the text and the rubric.
3. Compare the tags to be evaluated to the score descriptions in the rubric.
4. Pay close attention to small details that might impact the final score and form accurate associations between tags and pass criteria.
5. Write a detailed reasoning justifying your evaluation in a bullet point format.
6. The reasoning must summarize the overall strengths and weaknesses of the output while quoting exact phrases from the output wherever required.
7. Output a list of words or phrases that you believe are the most important in determining the score.
8. Assign a final score based on the scoring rubric.

Data to evaluate:
{data}

Pass Criteria:
{pass_criteria}

Rubric:
{rubric}

Your output must be in the following format:
<reasoning>
[Detailed reasoning justifying your evaluation in a bullet point format according to the specifics defined above]
</reasoning>
<highlight>
[List of words or phrases that you believe are the most important in determining the score]
</highlight>
<score>
[The final integer score assigned based on the scoring rubric]
</score>
"""

INPUT_OUTPUT_DATA_FORMAT = """
<INPUT>
{input_text}
</INPUT>

<OUTPUT>
{output_text}
</OUTPUT>
"""

INPUT_DATA_FORMAT = """
<INPUT>
{input_text}
</INPUT>
"""


class Glider(ThreeStageGuardrail[ChatMessages, str]):
    """A prompt based guardrail from Patronus AI that utilizes pass criteria and a rubric to judge text.

    For more information, see the model card:[GLIDER](https://huggingface.co/PatronusAI/glider). It outputs its reasoning,
    highlights for what determined the score, and an integer score.

    Args:
        pass_criteria: A question or description of what you are validating.
        rubric: A scoring rubric, describing to the model how to score the provided data.
        pass_threshold: The rubric score at which the text counts as passing. ``valid`` is
            ``rubric_score >= pass_threshold`` (or ``<=`` when ``higher_is_better`` is False).
        model_id: HuggingFace path to model.
        provider: Reserved for future extensibility. Currently unused.
        higher_is_better: Whether higher rubric scores mean better/passing text. Set to
            False for rubrics where higher scores mean worse text.
        score_range: Optional ``(min, max)`` bounds of the rubric scale. GLIDER's rubric is
            free text, so the bounds can't be inferred; supply them to get a normalized
            canonical risk in ``score``. When omitted, ``score`` is None and the raw rubric
            value is still available in ``extra["rubric_score"]``.

    Raise:
        ValueError: Can only use model path to GLIDER from HuggingFace.

    """

    SUPPORTED_MODELS: ClassVar = ["PatronusAI/glider"]

    def __init__(
        self,
        pass_criteria: str,
        rubric: str,
        pass_threshold: int,
        model_id: str | None = None,
        provider: StandardProvider | None = None,  # Reserved for future extensibility
        higher_is_better: bool = True,
        score_range: tuple[int, int] | None = None,
    ) -> None:
        """Initialize the GLIDER guardrail."""
        self.model_id = default(model_id, self.SUPPORTED_MODELS)
        self.pass_criteria = pass_criteria
        self.rubric = rubric
        self.pass_threshold = pass_threshold
        self.higher_is_better = higher_is_better
        self.score_range = score_range
        self.system_prompt = SYSTEM_PROMPT_GLIDER
        self.provider = provider  # Reserved for future extensibility
        self.model = pipeline("text-generation", self.model_id, max_new_tokens=2048, return_full_text=False)

    def validate(self, input_text: str, output_text: str | None = None) -> GuardrailOutput:  # type: ignore[override]
        """Use the provided pass criteria and rubric to judge the input and output text provided.

        Args:
            input_text: The initial text to evaluate.
            output_text: Optional subsequent text to evaluate alongside input.

        Returns:
            GuardrailOutput where ``valid`` maps the rubric score through
            ``pass_threshold``, ``explanation`` is the model's reasoning, and
            ``extra`` holds ``rubric_score`` and ``highlights``. When the rubric
            score cannot be parsed, the output fails closed (``valid=False``
            with ``extra={"parse_failure": True}``).

        """
        return self._execute(input_text, output_text)

    def _pre_processing(
        self, input_text: str, output_text: str | None = None
    ) -> GuardrailPreprocessOutput[ChatMessages]:
        if output_text is None:
            data = INPUT_DATA_FORMAT.format(input_text=input_text)
        else:
            data = INPUT_OUTPUT_DATA_FORMAT.format(input_text=input_text, output_text=output_text)
        prompt = self.system_prompt.format(data=data, pass_criteria=self.pass_criteria, rubric=self.rubric)
        return GuardrailPreprocessOutput(data=[{"role": "user", "content": prompt}])

    def _inference(self, message: GuardrailPreprocessOutput[ChatMessages]) -> GuardrailInferenceOutput[str]:
        """Run text-generation pipeline on chat messages."""
        generated_text: str = self.model(message.data)[0]["generated_text"]  # type: ignore[assignment]
        return GuardrailInferenceOutput(data=generated_text)

    def _post_processing(self, model_outputs: GuardrailInferenceOutput[str]) -> GuardrailOutput:
        generated_text = model_outputs.data
        score_match = SCORE_PATTERN.search(generated_text)
        if score_match is None:
            return GuardrailOutput(valid=False, explanation=generated_text, extra={"parse_failure": True})

        rubric_score = int(score_match.group(1))
        passed = rubric_score >= self.pass_threshold if self.higher_is_better else rubric_score <= self.pass_threshold
        reasoning_match = REASONING_PATTERN.search(generated_text)
        highlight_match = HIGHLIGHT_PATTERN.search(generated_text)
        # GLIDER's rubric is free text, so a canonical risk score is only
        # available when the caller supplies the scale via ``score_range``.
        score = (
            normalize_rubric_to_risk(
                rubric_score, self.score_range[0], self.score_range[1], higher_is_better=self.higher_is_better
            )
            if self.score_range is not None
            else None
        )
        return GuardrailOutput(
            valid=passed,
            score=score,
            explanation=reasoning_match.group(1) if reasoning_match else generated_text,
            extra={
                "rubric_score": rubric_score,
                "highlights": highlight_match.group(1) if highlight_match else None,
            },
        )
