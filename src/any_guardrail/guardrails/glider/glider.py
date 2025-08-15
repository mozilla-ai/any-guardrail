from typing import Any, ClassVar

from transformers import pipeline  # type: ignore [attr-defined]

from any_guardrail.guardrails.huggingface import HuggingFace
from any_guardrail.types import GuardrailOutput

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

DEFAULT_DATA_FORMAT = """
<INPUT>
{input_text}
</INPUT>

<OUTPUT>
{output_text}
</OUTPUT>
"""


class Glider(HuggingFace):
    """A prompt based guardrail from Patronus AI that utilizes pass criteria and a rubric to judge text.

    For more information, see the model card:[GLIDER](https://huggingface.co/PatronusAI/glider). It outputs its reasoning,
    highlights for what determined the score, and an integer score.

    Args:
        model_id: HuggingFace path to model.
        pass_criteria: A question or description of what you are validating.
        rubric: A scoring rubric, describing to the model how to score the provided data.

    Raise:
        ValueError: Can only use model path to GLIDER from HuggingFace.

    """

    SUPPORTED_MODELS: ClassVar = ["PatronusAI/glider"]

    def __init__(self, model_id: str, pass_criteria: str, rubric: str) -> None:
        """Initialize the GLIDER guardrail."""
        super().__init__(model_id)
        self.pass_criteria = pass_criteria
        self.rubric = rubric
        self.system_prompt = SYSTEM_PROMPT_GLIDER

    def validate(self, input_text: str, output_text: str) -> GuardrailOutput:  # type: ignore[override]
        """Use the provided pass criteria and rubric to judge the input and output text provided.

        Args:
            input_text: the initial text.
            output_text: the subsequent text.

        Returns:
            An explanation in the format provided by the system prompt.

        """
        message = self._pre_processing(input_text, output_text)
        result = self._inference(message)
        return GuardrailOutput(explanation=result)

    def _load_model(self) -> None:
        pipe = pipeline("text-classification", self.model_id)
        self.model = pipe

    def _pre_processing(self, input_text: str, output_text: str) -> list[dict[str, str]]:  # type: ignore[override]
        data = DEFAULT_DATA_FORMAT.format(input_text=input_text, output_text=output_text)
        prompt = self.system_prompt.format(data=data, pass_criteria=self.pass_criteria, rubric=self.rubric)
        return [{"role": "user", "content": prompt}]

    def _inference(self, message: list[dict[str, str]]) -> Any:
        return self.model(message)[0]["generated_text"]
