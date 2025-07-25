"""
Panoglin guardrail for detecting unsafe content using a discriminative model pipeline.
"""
from any_guardrail.guardrails.guardrail import Guardrail
from transformers import pipeline
from typing import Any
from ..utils.constants import LABEL_UNSAFE, LABEL_SAFE, LABEL_UNSAFE_LOWER

class Panoglin(Guardrail):
    """
    Guardrail for detecting unsafe content using a discriminative model pipeline.
    Args:
        modelpath (str): Path to the model.
    """
    def __init__(self, modelpath: str) -> None:
        """
        Initialize Panoglin with model path.
        """
        self.modelpath = modelpath
        try:
            self.model = self.model_instantiation()
        except Exception as e:
            raise RuntimeError(f"Failed to load model pipeline: {e}")

    def classify(self, input_text: str) -> str:
        """
        Classify input_text for unsafe content. Returns 'UNSAFE' or 'SAFE'.
        """
        try:
            classification = self.model(input_text)
            if classification[0]["label"] == LABEL_UNSAFE_LOWER:
                return LABEL_UNSAFE
            else:
                return LABEL_SAFE
        except Exception as e:
            raise RuntimeError(f"Error during classification: {e}")

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

