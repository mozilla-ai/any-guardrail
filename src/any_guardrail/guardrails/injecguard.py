from any_guardrail.guardrails.guardrail import Guardrail
from any_guardrail.types import GuardrailOutput, GuardrailModel
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, Pipeline

INJECGUARD_LABEL = "injection"


class InjecGuard(Guardrail):
    """
    Prompt injection detection encoder based model. For more information, please see the model card:
    [InjecGuard](https://huggingface.co/leolee99/InjecGuard)

    Args:
        model_id: HuggingFace path to model.

    Raises:
        ValueError: Can only use the model path for InjecGuard from HuggingFace
    """

    def __init__(self, model_id: str) -> None:
        super().__init__(model_id)
        if self.model_id in ["leolee99/InjecGuard"]:
            self.guardrail = self._model_instantiation()
        else:
            raise ValueError("Must use the following keyword argument to instantiate model: leolee99/InjecGuard")

    def validate(self, input_text: str) -> GuardrailOutput:
        """
        Classify some text to see if it contains a prompt injection attack.

        Args:
            input_text: the text to safety_review for prompt injection attacks
        Returns:
            True if there is a prompt injection attack, False otherwise
        """
        if isinstance(self.guardrail.model, Pipeline):
            classification = self.guardrail.model(input_text)
            return GuardrailOutput(unsafe=classification[0]["label"] == INJECGUARD_LABEL)
        else:
            raise TypeError("Using incorrect model type for InjecGuard.")

    def _model_instantiation(self) -> GuardrailModel:
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)  # type: ignore[no-untyped-call]
        model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
        pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
        return GuardrailModel(model=pipe)
