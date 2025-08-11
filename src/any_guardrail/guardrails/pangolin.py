from typing import ClassVar

from transformers import pipeline

from any_guardrail.guardrail import Guardrail
from any_guardrail.types import GuardrailOutput

PANGOLIN_INJECTION_LABEL = "unsafe"


class Pangolin(Guardrail):
    """Prompt injection detection encoder based models.

    For more information, please see the model card:
    [Pangolin Base](https://huggingface.co/dcarpintero/pangolin-guard-base)
    [Pangolin Large](https://huggingface.co/dcarpintero/pangolin-guard-large).

    Args:
        model_id: HuggingFace path to model.

    Raises:
        ValueError: Can only use model paths for Pangolin from HuggingFace

    """

    SUPPORTED_MODELS: ClassVar = ["dcarpintero/pangolin-guard-large", "dcarpintero/pangolin-guard-base"]

    def __init__(self, model_id: str) -> None:
        """Initialize the Pangolin guardrail."""
        super().__init__(model_id)

    def validate(self, input_text: str) -> GuardrailOutput:
        """Classify some text to see if it contains a prompt injection attack.

        Args:
            input_text: the text to validate for prompt injection attacks
        Returns:
            True if there is a prompt injection attack, False otherwise

        """
        classification = self._inference(input_text)
        return GuardrailOutput(unsafe=self._post_processing(classification))

    def _load_model(self) -> None:
        pipe = pipeline("text-classification", self.model_id)
        self.model = pipe

    def _inference(self, input_text: str) -> list[dict[str, str | float]]:
        return self.model(input_text)

    def _post_processing(self, classification: list[dict[str, str | float]]) -> bool:
        return classification[0]["label"] == PANGOLIN_INJECTION_LABEL
