from any_guardrail.guardrails.guardrail import Guardrail
from any_guardrail.utils.custom_types import ClassificationOutput, GuardrailModel
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, Pipeline  # type: ignore[attr-defined]

JASPER_INJECTION_LABEL = "INJECTION"


class Jasper(Guardrail):
    """
    Prompt injection detection encoder based models. For more information, please see the model card:
    https://huggingface.co/JasperLS/deberta-v3-base-injection
    https://huggingface.co/JasperLS/gelectra-base-injection
    Args:
        modelpath: HuggingFace path to model.

    Raises:
        ValueError: Can only use model paths for Jasper models from HuggingFace.
    """

    def __init__(self, modelpath: str) -> None:
        super().__init__(modelpath)
        if self.modelpath in ["JasperLS/deberta-v3-base-injection", "JasperLS/gelectra-base-injection"]:
            self.guardrail = self._model_instantiation()
        else:
            raise ValueError(
                "Must use one of the following keyword arguments to instantiate model: "
                "\n\n JasperLS/deberta-v3-base-injection \n JasperLS/gelectra-base-injection"
            )

    def classify(self, input_text: str) -> ClassificationOutput:
        """
        Classify some text to see if it contains a prompt injection attack.

        Args:
            input_text: the text to classify for prompt injection attacks
        Returns:
            True if there is a prompt injection attack, False otherwise
        """
        if isinstance(self.guardrail.model, Pipeline):
            classification = self.guardrail.model(input_text)
            return ClassificationOutput(unsafe=classification[0]["label"] == JASPER_INJECTION_LABEL)
        else:
            raise TypeError("Using incorrect model type for Jasper models.")

    def _model_instantiation(self) -> GuardrailModel:
        tokenizer = AutoTokenizer.from_pretrained(self.modelpath)  # type: ignore[no-untyped-call]
        model = AutoModelForSequenceClassification.from_pretrained(self.modelpath)
        pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
        return GuardrailModel(model=pipe)
