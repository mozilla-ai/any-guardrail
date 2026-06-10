from typing import ClassVar

from any_guardrail.base import GuardrailOutput, ThreeStageGuardrail
from any_guardrail.guardrails.utils import default
from any_guardrail.providers.base import StandardProvider
from any_guardrail.providers.huggingface import HuggingFaceProvider
from any_guardrail.types import AnyDict, CategoryResult, StandardInferenceOutput, StandardPreprocessOutput

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


class DuoGuard(ThreeStageGuardrail[AnyDict, AnyDict, bool, None, float]):
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
        self.provider = provider or HuggingFaceProvider(
            tokenizer_id=self.MODELS_TO_TOKENIZER[self.model_id],
            multi_label=True,
        )
        self.provider.load_model(self.model_id)
        # Qwen-family tokenizers ship without a pad token; HF needs one for padded batching.
        # Encoderfile bakes tokenizer config into the binary, so there's no tokenizer to touch.
        tokenizer = getattr(self.provider, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.pad_token = tokenizer.eos_token

    def _pre_processing(self, input_text: str) -> StandardPreprocessOutput:
        return self.provider.pre_process(input_text)

    def _inference(self, model_inputs: StandardPreprocessOutput) -> StandardInferenceOutput:
        return self.provider.infer(model_inputs)

    def _post_processing(self, model_outputs: StandardInferenceOutput) -> GuardrailOutput[bool, None, float]:
        # Providers expose per-category probabilities in ``scores``:
        # HuggingFaceProvider applies sigmoid when ``multi_label=True``; the
        # encoderfile binary applies sigmoid internally for multi-label heads.
        probabilities = [float(probability) for probability in model_outputs.data["scores"][0]]
        categories = [
            CategoryResult(name=category, score=probability, triggered=probability > self.threshold)
            for category, probability in zip(DUOGUARD_CATEGORIES, probabilities, strict=True)
        ]
        return GuardrailOutput(
            valid=not any(category.triggered for category in categories),
            score=max(probabilities),
            categories=categories,
        )
