from any_guardrail.guardrail import Guardrail
from any_guardrail.types import GuardrailOutput
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

PANGOLIN_INJECTION_LABEL = "unsafe"


class Pangolin(Guardrail):
    """
    Prompt injection detection encoder based models. For more information, please see the model card:
    [Pangolin Base](https://huggingface.co/dcarpintero/pangolin-guard-base)
    [Pangolin Large](https://huggingface.co/dcarpintero/pangolin-guard-large)

    Args:
        model_id: HuggingFace path to model.

    Raises:
        ValueError: Can only use model paths for Pangolin from HuggingFace
    """

    SUPPORTED_MODELS = ["dcarpintero/pangolin-guard-large", "dcarpintero/pangolin-guard-base"]

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
        return GuardrailOutput(unsafe=classification[0]["label"] == PANGOLIN_INJECTION_LABEL)

    def _load_model(self) -> None:
        model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        pipe = pipeline("text-classification", model = model, tokenizer = tokenizer)
        self.model = pipe
        self.tokenizer = tokenizer
