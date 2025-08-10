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
        if output_text:
            inputs = self.tokenizer(input_text, return_tensors="pt")
        else:
            inputs = self.tokenizer(input_text, output_text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            unsafe_prob = F.softmax(outputs.logits, dim=-1)[:, 1]
        final_score = unsafe_prob.item()

        return GuardrailOutput(unsafe=final_score > self.threshold, score=final_score)

    def _load_model(self) -> None:
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)  # type: ignore[no-untyped-call]
        model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
        model.eval()
        self.model = model
        self.tokenizer = tokenizer
