from typing import ClassVar

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from any_guardrail.guardrail import Guardrail
from any_guardrail.types import GuardrailOutput

PROTECTAI_INJECTION_LABEL = "INJECTION"


class Protectai(Guardrail):
    """Prompt injection detection encoder based models.

    For more information, please see the model cards:
    [ProtectA](https://huggingface.co/collections/protectai/llm-security-65c1f17a11c4251eeab53f40).

    Args:
        model_id: HuggingFace path to model.

    Raises:
        ValueError: Can only use model paths for ProtectAI from HuggingFace.

    """

    SUPPORTED_MODELS: ClassVar = [
        "protectai/deberta-v3-base-prompt-injection",
        "protectai/deberta-v3-small-prompt-injection-v2",
        "protectai/deberta-v3-base-prompt-injection-v2",
    ]

    def validate(self, input_text: str) -> GuardrailOutput:
        """Classify some text to see if it contains a prompt injection attack.

        Args:
            input_text: the text to validate for prompt injection attacks.

        Returns:
            True if there is a prompt injection attack, False otherwise.

        """
        classification = self._inference(input_text)
        return GuardrailOutput(unsafe=self._post_processing(classification))

    def _load_model(self) -> None:
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)  # type: ignore[no-untyped-call]
        model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
        pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
        self.model = pipe
        self.tokenizer = tokenizer

    def _inference(self, input_text: str) -> list[dict[str, str | float]]:
        return self.model(input_text)

    def _post_processing(self, classification: list[dict[str, str | float]]) -> bool:
        return classification[0]["label"] == PROTECTAI_INJECTION_LABEL
