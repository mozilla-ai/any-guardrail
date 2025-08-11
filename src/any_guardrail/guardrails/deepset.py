from typing import ClassVar

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from any_guardrail.guardrail import Guardrail
from any_guardrail.types import GuardrailOutput

DEEPSET_INJECTION_LABEL = "INJECTION"


class Deepset(Guardrail):
    """Wrapper for prompt injection detection model from Deepset.

    Please see model card for more information: [Deepset](https://huggingface.co/deepset/deberta-v3-base-injection).

    Args:
        model_id: HuggingFace path to model

    Raises:
        ValueError: Only supports Deepset models from HuggingFace

    """

    SUPPORTED_MODELS: ClassVar = ["deepset/deberta-v3-base-injection"]

    def validate(self, input_text: str) -> GuardrailOutput:
        """Classifies whether the provided text is a prompt injection attack or not.

        Args:
            input_text: the text that you want to check for prompt injection attacks
        Returns:
            True if there is a prompt injection attack, False otherwise

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
        return classification[0]["label"] == DEEPSET_INJECTION_LABEL
