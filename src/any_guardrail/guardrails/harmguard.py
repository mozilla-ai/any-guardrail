"""
Harmguard guardrail for detecting harmful content using a sequence classification model.
"""
from any_guardrail.guardrails.guardrail import Guardrail
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import torch
from typing import Any, Tuple
from ..utils.constants import DEFAULT_THRESHOLD, LABEL_UNSAFE, LABEL_SAFE

class Harmguard(Guardrail):
    """
    Guardrail for detecting harmful content using a sequence classification model.
    Args:
        modelpath (str): Path to the model.
    """
    def __init__(self, modelpath: str) -> None:
        """
        Initialize Harmguard with model path.
        """
        self.modelpath = modelpath
        try:
            self.model, self.tokenizer = self.model_instantiation()
        except Exception as e:
            raise RuntimeError(f"Failed to load model or tokenizer: {e}")
        self.threshold = DEFAULT_THRESHOLD # Taken from the Harmguard paper, but editable by the user

    def classify(self, input_text: str, output_text: str = None) -> Tuple[str, float]:
        """
        Classify input_text (and optionally output_text) for harmful content.
        Returns a tuple (label, score).
        """
        try:
            device = self.model.device
            if output_text == None:
                inputs = self.tokenizer(input_text, return_tensors="pt")
            else:
                inputs = self.tokenizer(input_text, output_text, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs)
                unsafe_prob = F.softmax(outputs.logits, dim=-1)[:, 1]
            final_score = unsafe_prob.item()
        except Exception as e:
            raise RuntimeError(f"Error during classification: {e}")

        if final_score > self.threshold:
            return LABEL_UNSAFE, final_score
        else:
            return LABEL_SAFE, final_score

    def model_instantiation(self) -> tuple[Any, Any]:
        """
        Load the model and tokenizer from the given model path.
        Returns:
            model, tokenizer
        """
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.modelpath)
            model = AutoModelForSequenceClassification.from_pretrained(self.modelpath)
            model.eval()
            return model, tokenizer
        except Exception as e:
            raise RuntimeError(f"Error loading model/tokenizer: {e}")
    