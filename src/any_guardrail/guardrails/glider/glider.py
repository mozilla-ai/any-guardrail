import re
from typing import ClassVar

from any_guardrail.base import GuardrailOutput, ThreeStageGuardrail
from any_guardrail.guardrails.utils import default
from any_guardrail.providers.base import StandardProvider
from any_guardrail.types import ChatMessages, GuardrailInferenceOutput, GuardrailPreprocessOutput

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


class Glider(ThreeStageGuardrail[ChatMessages, str, None, str, int | None]):
    """A prompt based guardrail from Patronus AI that utilizes pass criteria and a rubric to judge text.

    For more information, see the model card:[GLIDER](https://huggingface.co/PatronusAI/glider). It outputs its reasoning,
    highlights for what determined the score, and an integer score.

    Args:
        model_id: HuggingFace path to model.
        pass_criteria: A question or description of what you are validating.
        rubric: A scoring rubric, describing to the model how to score the provided data.
        provider: Reserved for future extensibility. Currently unused.

    Raise:
        ValueError: Can only use model path to GLIDER from HuggingFace.

    """

    SUPPORTED_MODELS: ClassVar = ["PatronusAI/glider"]

    def __init__(
        self,
        pass_criteria: str,
        rubric: str,
        model_id: str | None = None,
        provider: StandardProvider | None = None,  # Reserved for future extensibility
    ) -> None:
        """Initialize the GLIDER guardrail."""
        from transformers import pipeline

        self.model_id = default(model_id, self.SUPPORTED_MODELS)
        self.pass_criteria = pass_criteria
        self.rubric = rubric
        self.system_prompt = SYSTEM_PROMPT_GLIDER
        self.provider = provider  # Reserved for future extensibility
        self.model = pipeline("text-generation", self.model_id, max_new_tokens=2048, return_full_text=False)

    def validate(self, input_text: str, output_text: str | None = None) -> GuardrailOutput[None, str, int | None]:
        """Use the provided pass criteria and rubric to judge the input and output text provided.

        Args:
            input_text: the initial text.
            output_text: the subsequent text.

        Returns:
            An explanation in the format provided by the system prompt.

        """
        message = self._pre_processing(input_text, output_text)
        result = self._inference(message)
        return self._post_processing(result)

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

    def _post_processing(self, model_outputs: GuardrailInferenceOutput[str]) -> GuardrailOutput[None, str, int | None]:
        score = re.findall(r"<score>\n(\d+)\n</score>", model_outputs.data)
        if len(score) != 0 and score[0].isdigit():
            final_score = int(score[0])
        else:
            final_score = None

        return GuardrailOutput(explanation=model_outputs.data, score=final_score)
