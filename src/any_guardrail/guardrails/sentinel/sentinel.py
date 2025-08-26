from typing import Any, ClassVar

from any_guardrail.base import GuardrailOutput
from any_guardrail.guardrails.huggingface import HuggingFace, _match_injection_label

SENTINEL_INJECTION_LABEL = "jailbreak"


class Sentinel(HuggingFace):
    """Prompt injection detection encoder based model.

    For more information, please see the model card:

    - [Sentinel](https://huggingface.co/qualifire/prompt-injection-sentinel).
    """

    SUPPORTED_MODELS: ClassVar = ["qualifire/prompt-injection-sentinel"]

    def _post_processing(self, model_outputs: dict[str, Any]) -> GuardrailOutput:
        return _match_injection_label(model_outputs, SENTINEL_INJECTION_LABEL, self.model.config.id2label)
