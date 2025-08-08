from any_guardrail.guardrail import Guardrail
from any_guardrail.types import GuardrailOutput
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

SENTINEL_INJECTION_LABEL = "jailbreak"


class Sentinel(Guardrail):
    """
    Prompt injection detection encoder based model. For more information, please see the model card:
    [Sentinel](https://huggingface.co/qualifire/prompt-injection-sentinel)

    Args:
        model_id: HuggingFace path to model.

    Raises:
        ValueError: Can only use model path for Sentinel from HuggingFace.
    """

    SUPPORTED_MODELS = ["qualifire/prompt-injection-sentinel"]

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
        return GuardrailOutput(unsafe=classification[0]["label"] == SENTINEL_INJECTION_LABEL)

    def _load_model(self) -> None:
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)  # type: ignore[no-untyped-call]
        model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
        pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
        self.model = pipe
        self.tokenizer = tokenizer
