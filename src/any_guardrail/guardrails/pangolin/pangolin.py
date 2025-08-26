from typing import Any, ClassVar

from any_guardrail.base import GuardrailOutput
from any_guardrail.guardrails.huggingface import HuggingFace, _match_injection_label

PANGOLIN_INJECTION_LABEL = "unsafe"


class Pangolin(HuggingFace):
    """Prompt injection detection encoder based models.

    For more information, please see the model card:

    - [Pangolin Base](https://huggingface.co/dcarpintero/pangolin-guard-base)
    """

    SUPPORTED_MODELS: ClassVar = ["dcarpintero/pangolin-guard-base"]

    def _post_processing(self, model_outputs: dict[str, Any]) -> GuardrailOutput:
        return _match_injection_label(model_outputs, PANGOLIN_INJECTION_LABEL, self.model.config.id2label)
