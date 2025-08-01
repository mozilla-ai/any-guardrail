from any_guardrail.guardrails.guardrail import Guardrail
from any_guardrail.utils.custom_types import ClassificationOutput, GuardrailModel
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, Pipeline  # type: ignore[attr-defined]

DEEPSET_INJECTION_LABEL = "INJECTION"


class Deepset(Guardrail):
    """
    Wrapper for prompt injection detection model from Deepset. Please see model card for more information.
    https://huggingface.co/deepset/deberta-v3-base-injection

    Args:
        modelpath: HuggingFace path to model

    Raises:
        ValueError: Only supports Deepset models from HuggingFace
    """

    def __init__(self, modelpath: str) -> None:
        super().__init__(modelpath)
        if self.modelpath in ["deepset/deberta-v3-base-injection"]:
            self.guardrail = self._model_instantiation()
        else:
            raise ValueError(
                "Only supports deepset/deberta-v3-base-injection. Please use this path to instantiate model."
            )

    def classify(self, input_text: str) -> ClassificationOutput:
        """
        Classifies whether the provided text is a prompt injection attack or not.

        Args:
            input_text: the text that you want to check for prompt injection attacks
        Returns:
            True if there is a prompt injection attack, False otherwise
        """
        if isinstance(self.guardrail.model, Pipeline):
            classification = self.guardrail.model(input_text)
            return ClassificationOutput(unsafe=classification[0]["label"] == DEEPSET_INJECTION_LABEL)
        else:
            raise TypeError("Using incorrect model type for Deepset.")

    def _model_instantiation(self) -> GuardrailModel:
        tokenizer = AutoTokenizer.from_pretrained(self.modelpath)  # type: ignore[no-untyped-call]
        model = AutoModelForSequenceClassification.from_pretrained(self.modelpath)
        pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
        return GuardrailModel(model=pipe)
