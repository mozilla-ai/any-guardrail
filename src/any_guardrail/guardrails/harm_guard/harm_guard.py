from typing import ClassVar

from any_guardrail.base import GuardrailOutput, StandardGuardrail
from any_guardrail.guardrails.utils import _softmax, default
from any_guardrail.providers.base import StandardProvider
from any_guardrail.providers.huggingface import HuggingFaceProvider
from any_guardrail.types import (
    BinaryScoreOutput,
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

    def validate(self, input_text: str, output_text: str | None = None) -> BinaryScoreOutput:  # type: ignore[override]
        """Validate whether the input (and optionally output) text is safe.

        Args:
            input_text: The prompt/input text to evaluate.
            output_text: Optional response/output text. When provided, evaluates the
                safety of the response in context of the input.

        Returns:
            GuardrailOutput with valid=True if safe, valid=False if harmful.
            The score represents the unsafe probability (0.0 = safe, 1.0 = unsafe).

        """
        model_inputs = self._pre_processing(input_text, output_text)
        model_outputs = self._inference(model_inputs)
        return self._post_processing(model_outputs)

    def _pre_processing(self, input_text: str, output_text: str | None = None) -> StandardPreprocessOutput:
        """Tokenize input text and optionally output text.

        When output_text is provided, the tokenizer creates a text pair input
        suitable for evaluating the safety of the response.
        """
        if output_text is None:
            tokenized = self.provider.tokenizer(input_text, return_tensors="pt")  # type: ignore[attr-defined]
        else:
            tokenized = self.provider.tokenizer(input_text, output_text, return_tensors="pt")  # type: ignore[attr-defined]
        return GuardrailPreprocessOutput(data=tokenized)

    def _inference(self, model_inputs: StandardPreprocessOutput) -> StandardInferenceOutput:
        return self.provider.infer(model_inputs)

    def _post_processing(self, model_outputs: StandardInferenceOutput) -> BinaryScoreOutput:
        logits = model_outputs.data["logits"][0].numpy()
        scores = _softmax(logits)  # type: ignore[no-untyped-call]
        final_score = float(scores[1])  # scores[1] is the unsafe probability
        return GuardrailOutput(valid=final_score < self.threshold, score=final_score)
