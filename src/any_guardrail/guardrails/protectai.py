from any_guardrail.guardrails.guardrail import Guardrail
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification  # type: ignore[attr-defined]
from typing import Any

PROTECTAI_INJECTION_LABEL = "INJECTION"


class ProtectAI(Guardrail):
    """
    Prompt injection detection encoder based models. For more information, please see the model cards:
    https://huggingface.co/collections/protectai/llm-security-65c1f17a11c4251eeab53f40

    Args:
        modelpath (str): HuggingFace path to model.
    """

    def __init__(self, modelpath: str) -> None:
        super().__init__(modelpath)
        if self.modelpath in [
            "protectai/deberta-v3-base-prompt-injection",
            "protectai/deberta-v3-small-prompt-injection-v2",
            "protectai/deberta-v3-base-prompt-injection-v2",
        ]:
            self.model = self._model_instantiation()
        else:
            raise ValueError(
                "Must use one of the following keyword arguments to instantiate model: "
                "\n\n protectai/deberta-v3-base-prompt-injection \n protectai/deberta-v3-small-prompt-injection-v2 \n"
                "protectai/deberta-v3-base-prompt-injection-v2"
            )

    def classify(self, input_text: str) -> str:
        """
        Classify some text to see if it contains a prompt injection attack.

        Args:
            input_text: the text to classify for prompt injection attacks
        Returns:
            True if there is a prompt injection attack, False otherwise
        """
        classification = self.model(input_text)
        return classification[0]["label"] == PROTECTAI_INJECTION_LABEL

    def _model_instantiation(self) -> Any:
        tokenizer = AutoTokenizer.from_pretrained(self.modelpath)  # type: ignore[no-untyped-call]
        model = AutoModelForSequenceClassification.from_pretrained(self.modelpath)
        pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
        return pipe
