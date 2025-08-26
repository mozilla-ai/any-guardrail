from typing import Any, ClassVar

from any_guardrail.base import GuardrailOutput
from any_guardrail.guardrails.huggingface import HuggingFace, _softmax

HARMGUARD_DEFAULT_THRESHOLD = 0.5  # Taken from the HarmGuard paper


class HarmGuard(HuggingFace):
    """Prompt injection detection encoder based model.

    For more information, please see the model card:

    - [HarmGuard](https://huggingface.co/hbseong/HarmAug-Guard).
    """

    SUPPORTED_MODELS: ClassVar = ["hbseong/HarmAug-Guard"]

    def __init__(self, model_id: str | None = None, threshold: float = HARMGUARD_DEFAULT_THRESHOLD) -> None:
        """Initialize the HarmGuard guardrail."""
        super().__init__(model_id)
        self.threshold = threshold

    def _post_processing(self, model_outputs: dict[str, Any]) -> GuardrailOutput:
        logits = model_outputs["logits"][0].numpy()
        scores = _softmax(logits)  # type: ignore[no-untyped-call]
        final_score = float(scores[1])
        return GuardrailOutput(valid=final_score < self.threshold, score=final_score)
