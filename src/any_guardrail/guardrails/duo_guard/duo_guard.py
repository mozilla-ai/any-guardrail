from typing import Any, ClassVar

from any_guardrail.base import GuardrailOutput
from any_guardrail.guardrails.huggingface import HuggingFace

DUOGUARD_CATEGORIES = [
    "Violent crimes",
    "Non-violent crimes",
    "Sex-related crimes",
    "Child sexual exploitation",
    "Specialized advice",
    "Privacy",
    "Intellectual property",
    "Indiscriminate weapons",
    "Hate",
    "Suicide and self-harm",
    "Sexual content",
    "Jailbreak prompts",
]

DUOGUARD_DEFAULT_THRESHOLD = 0.5  # Taken from the DuoGuard model card.


class DuoGuard(HuggingFace):
    """Guardrail that classifies text based on the categories in DUOGUARD_CATEGORIES.

    For more information, please see the model card:

    - [DuoGuard](https://huggingface.co/collections/DuoGuard/duoguard-models-67a29ad8bd579a404e504d21).
    """

    SUPPORTED_MODELS: ClassVar = [
        "DuoGuard/DuoGuard-0.5B",
        "DuoGuard/DuoGuard-1B-Llama-3.2-transfer",
        "DuoGuard/DuoGuard-1.5B-transfer",
    ]

    MODELS_TO_TOKENIZER: ClassVar = {
        "DuoGuard/DuoGuard-0.5B": "Qwen/Qwen2.5-0.5B",
        "DuoGuard/DuoGuard-1B-Llama-3.2-transfer": "meta-llama/Llama-3.2-1B",
        "DuoGuard/DuoGuard-1.5B-transfer": "Qwen/Qwen2.5-1.5B",
    }

    def __init__(self, model_id: str | None = None, threshold: float = DUOGUARD_DEFAULT_THRESHOLD) -> None:
        """Initialize the DuoGuard model."""
        super().__init__(model_id)
        self.threshold = threshold

    def _load_model(self) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODELS_TO_TOKENIZER[self.model_id])  # type: ignore[no-untyped-call]
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def _post_processing(self, model_outputs: dict[str, Any]) -> GuardrailOutput:
        from torch.nn.functional import sigmoid

        probabilities = sigmoid(model_outputs["logits"][0]).tolist()
        predicted_labels = {
            category: prob > self.threshold for category, prob in zip(DUOGUARD_CATEGORIES, probabilities, strict=True)
        }
        return GuardrailOutput(
            valid=not any(predicted_labels.values()), explanation=predicted_labels, score=max(probabilities)
        )
