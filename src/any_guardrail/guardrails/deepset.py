from any_guardrail.guardrails.guardrail import Guardrail
from any_guardrail.types import GuardrailOutput, GuardrailModel
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, Pipeline

DEEPSET_INJECTION_LABEL = "INJECTION"


class Deepset(Guardrail):
    """
    Wrapper for prompt injection detection model from Deepset. Please see model card for more information:
    [Deepset](https://huggingface.co/deepset/deberta-v3-base-injection)

    Args:
        model_id: HuggingFace path to model

    Raises:
        ValueError: Only supports Deepset models from HuggingFace
    """

    def __init__(self, model_id: str) -> None:
        super().__init__(model_id)
        if self.model_id in ["deepset/deberta-v3-base-injection"]:
            self.guardrail = self._model_instantiation()
        else:
            raise ValueError(
                "Only supports deepset/deberta-v3-base-injection. Please use this path to instantiate model."
            )

    def validate(self, input_text: str) -> GuardrailOutput:
        """
        Classifies whether the provided text is a prompt injection attack or not.

        Args:
            input_text: the text that you want to check for prompt injection attacks
        Returns:
            True if there is a prompt injection attack, False otherwise
        """
        if isinstance(self.guardrail.model, Pipeline):
            classification = self.guardrail.model(input_text)
            return GuardrailOutput(unsafe=classification[0]["label"] == DEEPSET_INJECTION_LABEL)
        else:
            raise TypeError("Using incorrect model type for Deepset.")

    def _model_instantiation(self) -> GuardrailModel:
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)  # type: ignore[no-untyped-call]
        model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
        pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
        return GuardrailModel(model=pipe)
