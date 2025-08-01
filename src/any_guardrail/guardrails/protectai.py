from any_guardrail.guardrails.guardrail import Guardrail
from any_guardrail.utils.custom_types import ClassificationOutput, GuardrailModel
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, Pipeline  # type: ignore[attr-defined]

PROTECTAI_INJECTION_LABEL = "INJECTION"


class ProtectAI(Guardrail):
    """
    Prompt injection detection encoder based models. For more information, please see the model cards:
    https://huggingface.co/collections/protectai/llm-security-65c1f17a11c4251eeab53f40

    Args:
        model_identifier: HuggingFace path to model.

    Raises:
        ValueError: Can only use model paths for ProtectAI from HuggingFace.
    """

    def __init__(self, model_identifier: str) -> None:
        super().__init__(model_identifier)
        if self.model_identifier in [
            "protectai/deberta-v3-base-prompt-injection",
            "protectai/deberta-v3-small-prompt-injection-v2",
            "protectai/deberta-v3-base-prompt-injection-v2",
        ]:
            self.guardrail = self._model_instantiation()
        else:
            raise ValueError(
                "Must use one of the following keyword arguments to instantiate model: "
                "\n\n protectai/deberta-v3-base-prompt-injection \n protectai/deberta-v3-small-prompt-injection-v2 \n"
                "protectai/deberta-v3-base-prompt-injection-v2"
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
            return ClassificationOutput(unsafe=classification[0]["label"] == PROTECTAI_INJECTION_LABEL)
        else:
            raise TypeError("Using incorrect model type to call ProtectAI.")

    def _model_instantiation(self) -> GuardrailModel:
        tokenizer = AutoTokenizer.from_pretrained(self.model_identifier)  # type: ignore[no-untyped-call]
        model = AutoModelForSequenceClassification.from_pretrained(self.model_identifier)
        pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
        return GuardrailModel(model=pipe)
