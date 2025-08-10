from any_guardrail.guardrail import Guardrail
from any_guardrail.types import GuardrailOutput
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

PROTECTAI_INJECTION_LABEL = "INJECTION"


class Protectai(Guardrail):
    """
    Prompt injection detection encoder based models. For more information, please see the model cards:
    [ProtectA](https://huggingface.co/collections/protectai/llm-security-65c1f17a11c4251eeab53f40)

    Args:
        model_id: HuggingFace path to model.

    Raises:
        ValueError: Can only use model paths for ProtectAI from HuggingFace.
    """

    SUPPORTED_MODELS = [
        "protectai/deberta-v3-base-prompt-injection",
        "protectai/deberta-v3-small-prompt-injection-v2",
        "protectai/deberta-v3-base-prompt-injection-v2",
    ]

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
        classification = self.model(input_text)
        return GuardrailOutput(unsafe=classification[0]["label"] == PROTECTAI_INJECTION_LABEL)

    def _load_model(self) -> None:
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)  # type: ignore[no-untyped-call]
        model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
        pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
        self.model = pipe
        self.tokenizer = tokenizer
