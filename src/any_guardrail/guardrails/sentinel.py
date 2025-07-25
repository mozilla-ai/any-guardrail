"""
Sentinel guardrail for detecting jailbreak prompts using a sequence classification model.
"""
from any_guardrail.guardrails.guardrail import Guardrail
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from typing import Any
from ..utils.constants import LABEL_UNSAFE, LABEL_SAFE, LABEL_JAILBREAK

class Sentinel(Guardrail):
    """
    Guardrail for detecting jailbreak prompts using a sequence classification model.
    Args:
        modelpath (str): Path to the model.
    """
    def __init__(self, modelpath: str) -> None:
        """
        Initialize Sentinel with model path.
        """
        self.modelpath = modelpath
        try:
            self.model = self.model_instantiation()
        except Exception as e:
            raise RuntimeError(f"Failed to load model or tokenizer: {e}")

    def classify(self, input_text: str) -> str:
        """
        Classify input_text for jailbreak prompts. Returns 'UNSAFE' or 'SAFE'.
        """
        try:
            classification = self.model(input_text)
            if classification[0]["label"] == LABEL_JAILBREAK:
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
