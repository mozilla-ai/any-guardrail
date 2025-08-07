from any_guardrail.guardrails.guardrail import Guardrail
from any_guardrail.utils.custom_types import ClassificationOutput, GuardrailModel
from transformers import pipeline, Pipeline

PANGOLIN_INJECTION_LABEL = "unsafe"


class Pangolin(Guardrail):
    """
    Prompt injection detection encoder based models. For more information, please see the model card:
    [Pangolin Base](https://huggingface.co/dcarpintero/pangolin-guard-base)
    [Pangolin Large](https://huggingface.co/dcarpintero/pangolin-guard-large)

    Args:
        model_identifier: HuggingFace path to model.

    Raises:
        ValueError: Can only use model paths for Pangolin from HuggingFace
    """

    def __init__(self, model_identifier: str) -> None:
        super().__init__(model_identifier)
        if self.model_identifier in ["dcarpintero/pangolin-guard-large", "dcarpintero/pangolin-guard-base"]:
            self.guardrail = self._model_instantiation()
        else:
            raise ValueError(
                "Must use one of the following keyword arguments to instantiate model: "
                "\n\n dcarpintero/pangolin-guard-large \n dcarpintero/pangolin-guard-base"
            )

    def safety_review(self, input_text: str) -> ClassificationOutput:
        """
        Classify some text to see if it contains a prompt injection attack.

        Args:
            input_text: the text to safety_review for prompt injection attacks
        Returns:
            True if there is a prompt injection attack, False otherwise
        """
        if isinstance(self.guardrail.model, Pipeline):
            classification = self.guardrail.model(input_text)
            return ClassificationOutput(unsafe=classification[0]["label"] == PANGOLIN_INJECTION_LABEL)
        else:
            raise TypeError("Using incorrect model type for Pangolin.")

    def _model_instantiation(self) -> GuardrailModel:
        pipe = pipeline("text-classification", self.model_identifier)
        return GuardrailModel(model=pipe)
