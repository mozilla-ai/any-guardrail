from any_guardrail.guardrails.guardrail import Guardrail
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification  # type: ignore[attr-defined]
from typing import Any

DEEPSET_INJECTION_LABEL = "INJECTION"


class Deepset(Guardrail):
    """
    Wrapper for prompt injection detection model from Deepset. Please see model card for more information.
    https://huggingface.co/deepset/deberta-v3-base-injection

    Args:
        modelpath (str): HuggingFace path to model
    """

    def __init__(self, modelpath: str) -> None:
        super().__init__(modelpath)
        if self.modelpath in ["deepset/deberta-v3-base-injection"]:
            self.model = self._model_instantiation()
        else:
            raise ValueError(
                "Only supports deepset/deberta-v3-base-injection. Please use this path to instantiate model."
            )

    def classify(self, input_text: str) -> bool:
        """
        Classifies whether the provided text is a prompt injection attack or not.

        Args:
            input_text: the text that you want to check for prompt injection attacks
        Returns:
            True if there is a prompt injection attack, False otherwise
        """
        classification = self.model(input_text)
        return classification[0]["label"] == DEEPSET_INJECTION_LABEL

    def _model_instantiation(self) -> Any:
        """
        Creates the pipeline object for this model from HuggingFace.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.modelpath)  # type: ignore[no-untyped-call]
        model = AutoModelForSequenceClassification.from_pretrained(self.modelpath)
        pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
        return pipe
