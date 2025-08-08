from any_guardrail.guardrail import Guardrail
from any_guardrail.types import GuardrailOutput, GuardrailModel
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, Pipeline

JASPER_INJECTION_LABEL = "INJECTION"


class Jasper(Guardrail):
    """
    Prompt injection detection encoder based models. For more information, please see the model card:
    [Jasper Deberta](https://huggingface.co/JasperLS/deberta-v3-base-injection)
    [Jasper Gelectra](https://huggingface.co/JasperLS/gelectra-base-injection)

    Args:
        model_id: HuggingFace path to model.

    Raises:
        ValueError: Can only use model paths for Jasper models from HuggingFace.
    """

    SUPPORTED_MODELS = ["JasperLS/deberta-v3-base-injection", "JasperLS/gelectra-base-injection"]

    def __init__(self, model_id: str) -> None:
        super().__init__(model_id)

    def validate(self, input_text: str) -> GuardrailOutput:
        """
        Classify some text to see if it contains a prompt injection attack.

        Args:
            input_text: the text to validate for prompt injection attacks
        Returns:
            True if there is a prompt injection attack, False otherwise
        """
        if isinstance(self.guardrail.model, Pipeline):
            classification = self.guardrail.model(input_text)
            return GuardrailOutput(unsafe=classification[0]["label"] == JASPER_INJECTION_LABEL)
        else:
            raise TypeError("Using incorrect model type for Jasper models.")

    def _model_instantiation(self) -> GuardrailModel:
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)  # type: ignore[no-untyped-call]
        model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
        pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
        return GuardrailModel(model=pipe)
