from typing import ClassVar

from torch.nn.functional import sigmoid

from any_guardrail.base import GuardrailOutput, ThreeStageGuardrail
from any_guardrail.guardrails.utils import default
from any_guardrail.providers.base import StandardProvider
from any_guardrail.providers.huggingface import HuggingFaceProvider
from any_guardrail.types import AnyDict, StandardInferenceOutput, StandardPreprocessOutput

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


class DuoGuard(ThreeStageGuardrail[AnyDict, AnyDict, bool, dict[str, bool], float]):
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

    def __init__(
        self,
        model_id: str | None = None,
        threshold: float = DUOGUARD_DEFAULT_THRESHOLD,
        provider: StandardProvider | None = None,
    ) -> None:
        """Initialize the DuoGuard model."""
        self.model_id = default(model_id, self.SUPPORTED_MODELS)
        self.threshold = threshold
        self.provider = provider or HuggingFaceProvider(tokenizer_id=self.MODELS_TO_TOKENIZER[self.model_id])
        self.provider.load_model(self.model_id)
        self.provider.tokenizer.pad_token = self.provider.tokenizer.eos_token  # type: ignore[attr-defined]

    def _pre_processing(self, input_text: str) -> StandardPreprocessOutput:
        return self.provider.pre_process(input_text)

    def _inference(self, model_inputs: StandardPreprocessOutput) -> StandardInferenceOutput:
        return self.provider.infer(model_inputs)

    def _post_processing(self, model_outputs: StandardInferenceOutput) -> GuardrailOutput[bool, dict[str, bool], float]:
        probabilities = sigmoid(model_outputs.data["logits"][0]).tolist()
        predicted_labels = {
            category: prob > self.threshold for category, prob in zip(DUOGUARD_CATEGORIES, probabilities, strict=True)
        }
        return GuardrailOutput(
            valid=not any(predicted_labels.values()), explanation=predicted_labels, score=max(probabilities)
        )
