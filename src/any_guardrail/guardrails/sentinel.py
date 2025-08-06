from any_guardrail.guardrails.guardrail import Guardrail
from any_guardrail.utils.custom_types import ClassificationOutput, GuardrailModel
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, Pipeline

SENTINEL_INJECTION_LABEL = "jailbreak"


class Sentinel(Guardrail):
    """
    Prompt injection detection encoder based model. For more information, please see the model card:
    https://huggingface.co/qualifire/prompt-injection-sentinel

    Args:
        model_identifier: HuggingFace path to model.

    Raises:
        ValueError: Can only use model path for Sentinel from HuggingFace.
    """

    def __init__(self, model_identifier: str) -> None:
        super().__init__(model_identifier)
        if self.model_identifier in ["qualifire/prompt-injection-sentinel"]:
            self.guardrail = self._model_instantiation()
        else:
            raise ValueError(
                "Must use the following keyword argument to instantiate model: qualifire/prompt-injection-sentinel"
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
            return ClassificationOutput(unsafe=classification[0]["label"] == SENTINEL_INJECTION_LABEL)
        else:
            raise TypeError("Using incorrect model type for Sentinel.")

    def _model_instantiation(self) -> GuardrailModel:
        tokenizer = AutoTokenizer.from_pretrained(self.model_identifier)  # type: ignore[no-untyped-call]
        model = AutoModelForSequenceClassification.from_pretrained(self.model_identifier)
        pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
        return GuardrailModel(model=pipe)
