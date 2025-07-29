"""
Jasper guardrails for detecting injection attacks using different sequence classification models.
"""

from any_guardrail.guardrails.guardrail import Guardrail
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification  # type: ignore[attr-defined]
from typing import Any

JASPER_INJECTION_LABEL = "INJECTION"


class Jasper(Guardrail):
    """
    Prompt injection detection encoder based models. For more information, please see the model card:
    https://huggingface.co/JasperLS/deberta-v3-base-injection
    https://huggingface.co/JasperLS/gelectra-base-injection
    Args:
        modelpath (str): HuggingFace path to model.
    """

    def __init__(self, modelpath: str) -> None:
        """
        Initialize Jasper with model path.
        """
        super().__init__(modelpath)
        if self.modelpath in ["JasperLS/deberta-v3-base-injection", "JasperLS/gelectra-base-injection"]:
            self.model, self.tokenizer = self._model_instantiation()
        else:
            raise ValueError(
                "Must use one of the following keyword arguments to instantiate model: "
                "\n\n JasperLS/deberta-v3-base-injection \n JasperLS/gelectra-base-injection"
            )

    def classify(self, input_text: str) -> bool:
        """
        Classify some text to see if it contains a prompt injection attack.

        Args:
            input_text: the text to classify for prompt injection attacks
        Returns:
            True if there is a prompt injection attack, False otherwise
        """
        classification = self.model(input_text)
        return classification[0]["label"] == JASPER_INJECTION_LABEL

    def _model_instantiation(self) -> Any:
        tokenizer = AutoTokenizer.from_pretrained(self.modelpath)  # type: ignore[no-untyped-call]
        model = AutoModelForSequenceClassification.from_pretrained(self.modelpath)
        pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
        return pipe
