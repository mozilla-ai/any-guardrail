from typing import ClassVar

from any_guardrail.base import GuardrailOutput, StandardGuardrail
from any_guardrail.guardrails.utils import default
from any_guardrail.providers.base import StandardProvider
from any_guardrail.providers.huggingface import HuggingFaceProvider
from any_guardrail.types import (
    CategoryResult,
    GuardrailPreprocessOutput,
    StandardInferenceOutput,
    StandardPreprocessOutput,
)

HARMGUARD_DEFAULT_THRESHOLD = 0.5  # Taken from the HarmGuard paper


class HarmGuard(StandardGuardrail):
    """Safety and jailbreak detection model based on DeBERTa-v3-large.

    HarmAug-Guard classifies the safety of LLM conversations and detects jailbreak attempts.
    It can evaluate either a single prompt or a prompt + response pair.

    For more information, please see the model card:

    - [HarmAug-Guard](https://huggingface.co/hbseong/HarmAug-Guard).
    """

    SUPPORTED_MODELS: ClassVar = ["hbseong/HarmAug-Guard"]

    def __init__(
        self,
        model_id: str | None = None,
        threshold: float = HARMGUARD_DEFAULT_THRESHOLD,
        provider: StandardProvider | None = None,
    ) -> None:
        """Initialize the HarmGuard guardrail."""
        self.model_id = default(model_id, self.SUPPORTED_MODELS)
        self.threshold = threshold
        self.provider = provider or HuggingFaceProvider()
        self.provider.load_model(self.model_id)

    def validate(self, input_text: str, output_text: str | None = None) -> GuardrailOutput:  # type: ignore[override]
        """Validate whether the input (and optionally output) text is safe.

        Args:
            input_text: The prompt/input text to evaluate.
            output_text: Optional response/output text. When provided, evaluates the
                safety of the response in context of the input.

        Returns:
            GuardrailOutput with valid=True if safe, valid=False if harmful.
            The score represents the unsafe probability (0.0 = safe, 1.0 = unsafe).

        """
        return self._execute(input_text, output_text)

    def _pre_processing(self, input_text: str, output_text: str | None = None) -> StandardPreprocessOutput:
        """Tokenize input text and optionally output text.

        When output_text is provided, the tokenizer creates a text pair input
        suitable for evaluating the safety of the response.
        """
        if output_text is None:
            tokenized = self.provider.tokenizer(input_text, return_tensors="pt")  # type: ignore[attr-defined]
        else:
            tokenized = self.provider.tokenizer(input_text, output_text, return_tensors="pt")  # type: ignore[attr-defined]
        device = getattr(self.provider, "device", None)
        if device is not None:
            tokenized = tokenized.to(device)
        return GuardrailPreprocessOutput(data=tokenized)

    def _inference(self, model_inputs: StandardPreprocessOutput) -> StandardInferenceOutput:
        return self.provider.infer(model_inputs)

    # Index of the "unsafe" class when the model exposes no usable label names.
    # HarmAug-Guard ships no id2label, and its card defines softmax(logits)[1]
    # as the unsafe probability.
    _UNSAFE_INDEX = 1

    def _post_processing(self, model_outputs: StandardInferenceOutput) -> GuardrailOutput:
        scores_row = [float(probability) for probability in model_outputs.data["scores"][0]]
        # Prefer resolving the unsafe class by the provider's label list (the
        # reason infer() surfaces ``labels``); fall back to the documented column
        # for models like HarmAug-Guard that don't ship meaningful label names.
        labels = model_outputs.data.get("labels")
        unsafe_index = self._UNSAFE_INDEX
        if labels is not None:
            unsafe_index = next(
                (i for i, name in enumerate(labels) if name.lower() == "unsafe"),
                self._UNSAFE_INDEX,
            )
        safe_index = 1 - unsafe_index  # HarmGuard is a binary safe/unsafe classifier.

        final_score = scores_row[unsafe_index]
        triggered = final_score >= self.threshold
        return GuardrailOutput(
            valid=not triggered,
            score=final_score,
            categories=[
                CategoryResult(name="safe", score=scores_row[safe_index], triggered=not triggered),
                CategoryResult(name="unsafe", score=final_score, triggered=triggered),
            ],
        )
