from typing import Any, ClassVar

from any_guardrail.base import GuardrailOutput, ThreeStageGuardrail
from any_guardrail.providers.huggingface import HuggingFaceProvider
from any_guardrail.types import GuardrailInferenceOutput, GuardrailPreprocessOutput

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


class DuoGuard(ThreeStageGuardrail[dict[str, Any], dict[str, Any], bool, dict[str, bool], float]):
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
        provider: HuggingFaceProvider | None = None,
    ) -> None:
        """Initialize the DuoGuard model."""
        self.model_id = model_id or self.SUPPORTED_MODELS[0]
        if self.model_id not in self.SUPPORTED_MODELS:
            msg = f"Only supports {self.SUPPORTED_MODELS}. Please use this path to instantiate model."
            raise ValueError(msg)
        self.threshold = threshold
        self.provider = provider or HuggingFaceProvider(tokenizer_id=self.MODELS_TO_TOKENIZER[self.model_id])
        self.provider.load_model(self.model_id)
        self.provider.tokenizer.pad_token = self.provider.tokenizer.eos_token

    def validate(self, input_text: str) -> GuardrailOutput[bool, dict[str, bool], float]:
        """Validate whether the input text is safe or not."""
        model_inputs = self._pre_processing(input_text)
        model_outputs = self._inference(model_inputs)
        return self._post_processing(model_outputs)

    def _pre_processing(self, input_text: str) -> GuardrailPreprocessOutput[dict[str, Any]]:
        return self.provider.pre_process(input_text)

    def _inference(
        self, model_inputs: GuardrailPreprocessOutput[dict[str, Any]]
    ) -> GuardrailInferenceOutput[dict[str, Any]]:
        return self.provider.infer(model_inputs)

    def _post_processing(
        self, model_outputs: GuardrailInferenceOutput[dict[str, Any]]
    ) -> GuardrailOutput[bool, dict[str, bool], float]:
        from torch.nn.functional import sigmoid

        probabilities = sigmoid(model_outputs.data["logits"][0]).tolist()
        predicted_labels = {
            category: prob > self.threshold for category, prob in zip(DUOGUARD_CATEGORIES, probabilities, strict=True)
        }
        return GuardrailOutput(
            valid=not any(predicted_labels.values()), explanation=predicted_labels, score=max(probabilities)
        )
