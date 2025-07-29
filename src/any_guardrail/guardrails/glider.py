"""
GLIDER guardrail for rubric-based evaluation of text outputs.
"""

from any_guardrail.guardrails.guardrail import Guardrail
from transformers import pipeline  # type: ignore[attr-defined]
from typing import Any

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


class GLIDER(Guardrail):
    """
    A prompt based guardrail from Patronus AI that utilizes pass criteria and a rubric to judge text. It outputs its reasoning,
    highlights for what determined the score, and an integer score. For more information, see the model card:
    https://huggingface.co/PatronusAI/glider
    Args:
        modelpath (str): HuggingFace path to model.
        pass_criteria (str): A question or description of what you are classifying.
        rubric (str): A scoring rubric, describing to the model how to score the provided data.
    """

    def __init__(self, modelpath: str, pass_criteria: str, rubric: str) -> None:
        """
        Initialize GLIDER with model path, pass criteria, and rubric.
        """
        super().__init__(modelpath)
        if self.modelpath in ["PatronusAI/glider"]:
            self.model = self._model_instantiation()
        else:
            raise ValueError("You must use the following model path: PatronusAI/glider")
        self.pass_criteria = pass_criteria
        self.rubric = rubric
        self.system_prompt = SYSTEM_PROMPT_GLIDER

    def classify(self, input_text: str, output_text: str) -> str:
        """
        Uses the provided pass criteria and rubric to just the input and output text provided.

        Args:
            input_text: the initial text
            output_text: the subsequent text
        Returns:
            A string in the format provided by the system prompt
        """
        data = """
            <INPUT>
            {input_text}
            </INPUT>

            <OUTPUT>
            {output_text}
            </OUTPUT>
            """.format(input_text=input_text, output_text=output_text)

        prompt = self.system_prompt.format(data=data, pass_criteria=self.pass_criteria, rubric=self.rubric)

        message = [{"role": "user", "content": prompt}]

        result = self.model(message)
        return result

    def _model_instantiation(self) -> Any:
        pipe = pipeline("text-classification", self.modelpath)
        return pipe
