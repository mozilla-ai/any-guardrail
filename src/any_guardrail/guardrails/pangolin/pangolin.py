from typing import Any, ClassVar

from any_guardrail.base import GuardrailOutput
from any_guardrail.guardrails.huggingface import HuggingFace, _match_injection_label
from any_guardrail.types import GuardrailInferenceOutput

PANGOLIN_INJECTION_LABEL = "unsafe"


class Pangolin(HuggingFace[dict[str, Any], dict[str, Any], bool, None, float]):
    """Prompt injection detection encoder based models.

    For more information, please see the model card:

    - [Pangolin Base](https://huggingface.co/dcarpintero/pangolin-guard-base)
    """

    SUPPORTED_MODELS: ClassVar = ["dcarpintero/pangolin-guard-base"]

    def _post_processing(
        self, model_outputs: GuardrailInferenceOutput[dict[str, Any]]
    ) -> GuardrailOutput[bool, None, float]:
        return _match_injection_label(model_outputs, PANGOLIN_INJECTION_LABEL, self.model.config.id2label)
