from any_guardrail.guardrails.guardrail import Guardrail
from any_guardrail.utils.custom_types import ClassificationOutput, GuardrailModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, PreTrainedModel
import torch.nn.functional as F
import torch

HARMGUARD_DEFAULT_THRESHOLD = 0.5  # Taken from the HarmGuard paper


class HarmGuard(Guardrail):
    """
    Prompt injection detection encoder based model. For more information, please see the model card:
    [HarmGuard](https://huggingface.co/hbseong/HarmAug-Guard)

    Args:
        model_identifier: HuggingFace path to model.

    Raises:
        ValueError: Can only use model path for HarmGuard from HuggingFace
    """

    def __init__(self, model_identifier: str, threshold: float = HARMGUARD_DEFAULT_THRESHOLD) -> None:
        super().__init__(model_identifier)
        if self.model_identifier in ["hbseong/HarmAug-Guard"]:
            self.guardrail = self._model_instantiation()
        else:
            raise ValueError("Must use the following keyword argument to instantiate model: hbseong/HarmAug-Guard")
        self.threshold = threshold

    def safety_review(self, input_text: str, output_text: str = "") -> ClassificationOutput:
        """
        Classifies input text and, optionally, output text for prompt injection detection.

        Args:
            input_text: the initial text that you want to safety_review
            output_text: the subsequent text that you want to safety_review

        Returns:
            True if it is a prompt injection attack, False otherwise, and the associated final score.
        """
        if self.guardrail.tokenizer:
            if output_text:
                inputs = self.guardrail.tokenizer(input_text, return_tensors="pt")
            else:
                inputs = self.guardrail.tokenizer(input_text, output_text, return_tensors="pt")
            if isinstance(self.guardrail.model, PreTrainedModel):
                with torch.no_grad():
                    outputs = self.guardrail.model(**inputs)
                    unsafe_prob = F.softmax(outputs.logits, dim=-1)[:, 1]
                final_score = unsafe_prob.item()
            else:
                raise TypeError("Using incorrect model type for HarmGuard.")

            return ClassificationOutput(unsafe=final_score > self.threshold, score=final_score)
        else:
            raise TypeError("Did not instantiate tokenizer.")

    def _model_instantiation(self) -> GuardrailModel:
        tokenizer = AutoTokenizer.from_pretrained(self.model_identifier)  # type: ignore[no-untyped-call]
        model = AutoModelForSequenceClassification.from_pretrained(self.model_identifier)
        model.eval()
        return GuardrailModel(model=model, tokenizer=tokenizer)
