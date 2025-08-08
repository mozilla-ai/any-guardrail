from any_guardrail.guardrail import Guardrail
from any_guardrail.types import ClassificationOutput, GuardrailModel
from transformers import pipeline, Pipeline

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
    [GLIDER](https://huggingface.co/PatronusAI/glider)

    Args:
        model_id: HuggingFace path to model.
        pass_criteria: A question or description of what you are safety_reviewing.
        rubric: A scoring rubric, describing to the model how to score the provided data.

    Raise:
        ValueError: Can only use model path to GLIDER from HuggingFace.
    """

    def __init__(self, model_id: str, pass_criteria: str, rubric: str) -> None:
        super().__init__(model_id)
        if self.model_id in ["PatronusAI/glider"]:
            self.guardrail = self._model_instantiation()
        else:
            raise ValueError("You must use the following model path: PatronusAI/glider")
        self.pass_criteria = pass_criteria
        self.rubric = rubric
        self.system_prompt = SYSTEM_PROMPT_GLIDER

    def safety_review(self, input_text: str, output_text: str) -> ClassificationOutput:
        """
        Uses the provided pass criteria and rubric to just the input and output text provided.

        Args:
            input_text: the initial text
            output_text: the subsequent text
        Returns:
            An explanation in the format provided by the system prompt
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
        if isinstance(self.guardrail.model, Pipeline):
            result = self.guardrail.model(message)
            return ClassificationOutput(explanation=result)
        else:
            raise TypeError("Using incorrect model type for GLIDER.")

    def _model_instantiation(self) -> GuardrailModel:
        pipe = pipeline("text-classification", self.model_id)
        return GuardrailModel(model=pipe)
