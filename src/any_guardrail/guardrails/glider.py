"""
Glider guardrail for rubric-based evaluation of text outputs.
"""
from any_guardrail.guardrails.guardrail import Guardrail
from transformers import pipeline
from typing import Any
from ..utils.constants import SYSTEM_PROMPT_GLIDER

class Glider(Guardrail):
    """
    Guardrail for rubric-based evaluation of text outputs.
    Args:
        modelpath (str): Path to the model.
        pass_criteria (str): Criteria for passing.
        rubric (str): Rubric for scoring.
    """
    def __init__(self, modelpath: str, pass_criteria: str, rubric: str) -> None:
        """
        Initialize Glider with model path, pass criteria, and rubric.
        """
        self.modelpath = modelpath
        try:
            self.model = self.model_instantiation()
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
        self.pass_criteria = pass_criteria
        self.rubric = rubric
        self.system_prompt = SYSTEM_PROMPT_GLIDER

    def classify(self, input_text: str, output_text: str) -> Any:
        """
        Evaluate input and output text using the rubric and return the result.
        """
        try:
            data = """
                <INPUT>
                {input_text}
                </INPUT>

                <OUTPUT>
                {output_text}
                </OUTPUT>
                """.format(input_text=input_text, output_text=output_text)
        
        
            prompt = self.system_prompt.format(data=data, pass_criteria=self.pass_criteria, rubric=self.rubric)

            message = [
                {"role": "user", "content": prompt}
            ]

            result = self.model(message) #TODO: make this into a JSON output
            return result
        except Exception as e:
            raise RuntimeError(f"Error during evaluation: {e}")

    def model_instantiation(self) -> Any:
        """
        Load the model pipeline from the given model path.
        Returns:
            pipeline object
        """
        try:
            pipe = pipeline("text-classification", self.modelpath)
            return pipe
        except Exception as e:
            raise RuntimeError(f"Error loading model pipeline: {e}")