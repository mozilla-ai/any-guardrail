"""
Duoguard guardrail for multi-category safety classification using a sequence classification model.
"""
from any_guardrail.guardrails.guardrail import Guardrail
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Any, Tuple, Dict
from ..utils.constants import DUOGUARD_CATEGORIES, DEFAULT_THRESHOLD, LABEL_UNSAFE, LABEL_SAFE

class Duoguard(Guardrail):
    """
    Guardrail for multi-category safety classification using a sequence classification model.
    Args:
        modelpath (str): Path to the model.
    """
    def __init__(self, modelpath: str) -> None:
        """
        Initialize Duoguard with model path.
        """
        self.modelpath = modelpath
        try:
            self.model, self.tokenizer = self.model_instantiation()
        except Exception as e:
            raise RuntimeError(f"Failed to load model or tokenizer: {e}")
        self.threshold = 0.5 # Taken from the Duoguard Model Card, but editable by the user

    def classify(self, input_text: str, output_text: str = None) -> Tuple[str, Dict[str, str]]:
        """
        Classify input_text (and optionally output_text) for multiple safety categories.
        Returns a tuple (overall_label, predicted_labels).
        """
        try:
            inputs = self.tokenizer(
                        input_text,
                        return_tensors="pt", 
                    )
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.sigmoid(logits)
            #TODO: Move this to a constants file
            categories = DUOGUARD_CATEGORIES
            prob_vector = probabilities[0].tolist()
            predicted_labels = {}
            for cat_name, prob in zip(categories, prob_vector):
                label = LABEL_UNSAFE if prob > self.threshold else LABEL_SAFE
                predicted_labels[cat_name] = label
            max_prob = max(prob_vector)
            overall_label = LABEL_UNSAFE if max_prob > self.threshold else LABEL_SAFE
        except Exception as e:
            raise RuntimeError(f"Error during classification: {e}")
        return overall_label, predicted_labels

    def model_instantiation(self) -> tuple[Any, Any]:
        """
        Load the model and tokenizer from the given model path.
        Returns:
            model, tokenizer
        """
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.modelpath)
            tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForSequenceClassification.from_pretrained(self.modelpath)
            return model, tokenizer
        except Exception as e:
            raise RuntimeError(f"Error loading model/tokenizer: {e}")