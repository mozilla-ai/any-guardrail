from typing import Tuple

from any_guardrail.guardrail import Guardrail
from any_guardrail.types import GuardrailOutput
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import torch

HARMGUARD_DEFAULT_THRESHOLD = 0.5  # Taken from the HarmGuard paper


class HarmGuard(Guardrail):
    """
    Prompt injection detection encoder based model. For more information, please see the model card:
    [HarmGuard](https://huggingface.co/hbseong/HarmAug-Guard)

    Args:
        model_id: HuggingFace path to model.

    Raises:
        ValueError: Can only use model path for HarmGuard from HuggingFace
    """

    SUPPORTED_MODELS = ["hbseong/HarmAug-Guard"]

    def __init__(self, model_id: str, threshold: float = HARMGUARD_DEFAULT_THRESHOLD) -> None:
        super().__init__(model_id)
        self.threshold = threshold

    def validate(self, input_text: str, output_text: str = "") -> GuardrailOutput:
        """
        Classifies input text and, optionally, output text for prompt injection detection.

        Args:
            input_text: the initial text that you want to validate
            output_text: the subsequent text that you want to validate

        Returns:
            True if it is a prompt injection attack, False otherwise, and the associated final score.
        """
        inputs = self._pre_processing(input_text, output_text)
        outputs = self._inference(inputs)
        final_unsafe, final_score = self._post_processing(outputs)

        return GuardrailOutput(unsafe=final_unsafe, score=final_score)

    def _load_model(self) -> None:
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)  # type: ignore[no-untyped-call]
        model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
        model.eval()
        self.model = model
        self.tokenizer = tokenizer

    def _pre_processing(self, input_text: str, output_text: str) -> torch.Tensor:
        if output_text:
            return self.tokenizer(input_text, return_tensors="pt")
        else:
            return self.tokenizer(input_text, output_text, return_tensors="pt")

    def _inference(self, inputs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs

    def _post_processing(self, outputs: torch.Tensor) -> Tuple[bool, float]:
        unsafe_prob = F.softmax(outputs.logits, dim=-1)[:, 1]
        final_score = unsafe_prob.item()
        return final_score > self.threshold, final_score
