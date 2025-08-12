from typing import ClassVar

from any_guardrail.guardrails.huggingface import HuggingFace, _match_injection_label
from any_guardrail.types import GuardrailOutput

SENTINEL_INJECTION_LABEL = "jailbreak"


class Sentinel(HuggingFace):
    """Prompt injection detection encoder based model.

    For more information, please see the model card:

    - [Sentinel](https://huggingface.co/qualifire/prompt-injection-sentinel).
    """

    SUPPORTED_MODELS: ClassVar = ["qualifire/prompt-injection-sentinel"]

    def _post_processing(self, model_outputs: list[dict[str, str | float]]) -> GuardrailOutput:
        return _match_injection_label(model_outputs, SENTINEL_INJECTION_LABEL, self.model.config.id2label)
