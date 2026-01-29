from typing import Any, ClassVar

from any_guardrail.base import GuardrailOutput, ThreeStageGuardrail
from any_guardrail.guardrails.utils import _softmax
from any_guardrail.providers.huggingface import HuggingFaceProvider
from any_guardrail.types import GuardrailInferenceOutput, GuardrailPreprocessOutput

HARMGUARD_DEFAULT_THRESHOLD = 0.5  # Taken from the HarmGuard paper


class HarmGuard(ThreeStageGuardrail[dict[str, Any], dict[str, Any], bool, None, float]):
    """Prompt injection detection encoder based model.

    For more information, please see the model card:

    - [HarmGuard](https://huggingface.co/hbseong/HarmAug-Guard).
    """

    SUPPORTED_MODELS: ClassVar = ["hbseong/HarmAug-Guard"]

    def __init__(
        self,
        model_id: str | None = None,
        threshold: float = HARMGUARD_DEFAULT_THRESHOLD,
        provider: HuggingFaceProvider | None = None,
    ) -> None:
        """Initialize the HarmGuard guardrail."""
        self.model_id = model_id or self.SUPPORTED_MODELS[0]
        if self.model_id not in self.SUPPORTED_MODELS:
            msg = f"Only supports {self.SUPPORTED_MODELS}. Please use this path to instantiate model."
            raise ValueError(msg)
        self.threshold = threshold
        self.provider = provider or HuggingFaceProvider()
        self.provider.load_model(self.model_id)

    def validate(self, input_text: str) -> GuardrailOutput[bool, None, float]:
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
    ) -> GuardrailOutput[bool, None, float]:
        logits = model_outputs.data["logits"][0].numpy()
        scores = _softmax(logits)  # type: ignore[no-untyped-call]
        final_score = float(scores[1])
        return GuardrailOutput(valid=final_score < self.threshold, score=final_score)
