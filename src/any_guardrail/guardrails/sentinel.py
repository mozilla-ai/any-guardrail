from any_guardrail.guardrails.guardrail import Guardrail
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification  # type: ignore[attr-defined]
from typing import Any

SENTINEL_INJECTION_LABEL = "jailbreak"


class Sentinel(Guardrail):
    """
    Prompt injection detection encoder based model. For more information, please see the model card:
    https://huggingface.co/qualifire/prompt-injection-sentinel

    Args:
        modelpath (str): HuggingFace path to model.
    """

    def __init__(self, modelpath: str) -> None:
        super().__init__(modelpath)
        if self.modelpath in ["qualifire/prompt-injection-sentinel"]:
            self.model, self.tokenizer = self._model_instantiation()
        else:
            raise ValueError(
                "Must use the following keyword argument to instantiate model: qualifire/prompt-injection-sentinel"
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
        return classification[0]["label"] == SENTINEL_INJECTION_LABEL

    def _model_instantiation(self) -> Any:
        tokenizer = AutoTokenizer.from_pretrained(self.modelpath)  # type: ignore[no-untyped-call]
        model = AutoModelForSequenceClassification.from_pretrained(self.modelpath)
        pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
        return pipe
