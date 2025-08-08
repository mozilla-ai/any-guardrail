from any_guardrail.guardrails.guardrail import Guardrail
from any_guardrail.types import GuardrailOutput, GuardrailModel
from transformers import pipeline, Pipeline

PANGOLIN_INJECTION_LABEL = "unsafe"


class Pangolin(Guardrail):
    """
    Prompt injection detection encoder based models. For more information, please see the model card:
    [Pangolin Base](https://huggingface.co/dcarpintero/pangolin-guard-base)
    [Pangolin Large](https://huggingface.co/dcarpintero/pangolin-guard-large)

    Args:
        model_id: HuggingFace path to model.

    Raises:
        ValueError: Can only use model paths for Pangolin from HuggingFace
    """

    SUPPORTED_MODELS = ["dcarpintero/pangolin-guard-large", "dcarpintero/pangolin-guard-base"]

    def __init__(self, model_id: str) -> None:
        super().__init__(model_id)

    def validate(self, input_text: str) -> GuardrailOutput:
        """
        Classify some text to see if it contains a prompt injection attack.

        Args:
            input_text: the text to safety_review for prompt injection attacks
        Returns:
            True if there is a prompt injection attack, False otherwise
        """
        if isinstance(self.guardrail.model, Pipeline):
            classification = self.guardrail.model(input_text)
            return GuardrailOutput(unsafe=classification[0]["label"] == PANGOLIN_INJECTION_LABEL)
        else:
            raise TypeError("Using incorrect model type for Pangolin.")

    def _model_instantiation(self) -> GuardrailModel:
        pipe = pipeline("text-classification", self.model_id)
        return GuardrailModel(model=pipe)
