from any_guardrail.guardrails.guardrail import Guardrail
from transformers import pipeline # type: ignore[attr-defined]
from typing import Any

PANGOLIN_INJECTION_LABEL = "UNSAFE"

class Pangolin(Guardrail):
    """
    Prompt injection detection encoder based models. For more information, please see the model card: 
    https://huggingface.co/dcarpintero/pangolin-guard-base
    https://huggingface.co/dcarpintero/pangolin-guard-large

    Args:
        modelpath (str): HuggingFace path to model.
    """
    def __init__(self, modelpath: str) -> None:
        super().__init__(modelpath)
        if self.modelpath in ["dcarpintero/pangolin-guard-large", "dcarpintero/pangolin-guard-base"]:
            self.model = self._model_instantiation()
        else:
            raise ValueError("Must use one of the following keyword arguments to instantiate model: " \
            "\n\n dcarpintero/pangolin-guard-large \n dcarpintero/pangolin-guard-base")

    def classify(self, input_text: str) -> bool:
        """
        Classify some text to see if it contains a prompt injection attack.

        Args:
            input_text: the text to classify for prompt injection attacks
        Returns:
            True if there is a prompt injection attack, False otherwise
        """
        classification = self.model(input_text)
        return classification[0]["label"] == PANGOLIN_INJECTION_LABEL

    def _model_instantiation(self) -> Any:
        print(self.modelpath)
        pipe = pipeline("text-classification", self.modelpath)
        return pipe

