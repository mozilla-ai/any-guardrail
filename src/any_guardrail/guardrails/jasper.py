"""
Jasper guardrails for detecting injection attacks using different sequence classification models.
"""
from typing import Any
from any_guardrail.guardrails.guardrail import Guardrail
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from ..utils.constants import LABEL_UNSAFE, LABEL_SAFE, LABEL_INJECTION_UPPER

class Jasper(Guardrail):
    """
    Guardrail for detecting injection attacks using the Gelectra model.
    Args:
        modelpath (str): Path to the model.
    """
    def __init__(self, modelpath: str) -> None:
        """
        Initialize Jasper with model path.
        """
        self.modelpath = modelpath
        try:
            self.model = self.model_instantiation()
        except Exception as e:
            raise RuntimeError(f"Failed to load model or tokenizer: {e}")

    def classify(self, input_text: str) -> str:
        """
        Classify input_text for injection attacks. Returns 'UNSAFE' or 'SAFE'.
        """
        try:
            classification = self.model(input_text)
            if classification[0]["label"] == LABEL_INJECTION_UPPER:
                return LABEL_UNSAFE
            else:
                return LABEL_SAFE
        except Exception as e:
            raise RuntimeError(f"Error during classification: {e}")

    def model_instantiation(self) -> Any:
        """
        Load the model and tokenizer from the given model path.
        Returns:
            pipeline object
        """
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.modelpath)
            model = AutoModelForSequenceClassification.from_pretrained(self.modelpath)
            pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
            return pipe
        except Exception as e:
            raise RuntimeError(f"Error loading model/tokenizer: {e}")