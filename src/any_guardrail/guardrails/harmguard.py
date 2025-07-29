from any_guardrail.guardrails.guardrail import Guardrail
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import torch
from typing import Any, Tuple

HARMGUARD_DEFAULT_THRESHOLD = 0.5  # Taken from the HarmGuard paper


class HarmGuard(Guardrail):
    """
    Prompt injection detection encoder based model. For more information, please see the model card:
    https://huggingface.co/hbseong/HarmAug-Guard
    Args:
        modelpath (str): HuggingFace path to model.
    """

    def __init__(self, modelpath: str, threshold: float = HARMGUARD_DEFAULT_THRESHOLD) -> None:
        super().__init__(modelpath)
        if self.modelpath in ["hbseong/HarmAug-Guard"]:
            self.model, self.tokenizer = self._model_instantiation()
        else:
            raise ValueError("Must use the following keyword argument to instantiate model: hbseong/HarmAug-Guard")
        self.threshold = threshold

    def classify(self, input_text: str, output_text: str = "") -> Tuple[bool, float]:
        """
        Classifies input text and, optionally, output text for prompt injection detection.

        Args:
            input_text: the initial text that you want to classify
            output_text: the subsequent text that you want to classify

        Returns:
            True if it is a prompt injection attack, False otherwise, and the associated final score.
        """
        if output_text:
            inputs = self.tokenizer(input_text, return_tensors="pt")
        else:
            inputs = self.tokenizer(input_text, output_text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            unsafe_prob = F.softmax(outputs.logits, dim=-1)[:, 1]
        final_score = unsafe_prob.item()

        return final_score > self.threshold, final_score

    def _model_instantiation(self) -> tuple[Any, Any]:
        tokenizer = AutoTokenizer.from_pretrained(self.modelpath)  # type: ignore[no-untyped-call]
        model = AutoModelForSequenceClassification.from_pretrained(self.modelpath)
        model.eval()
        return model, tokenizer
