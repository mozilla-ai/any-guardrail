from typing import Any, ClassVar

from any_guardrail.base import GuardrailOutput
from any_guardrail.guardrails.huggingface import HuggingFace, _match_injection_label
from any_guardrail.types import GuardrailInferenceOutput

DEEPSET_INJECTION_LABEL = "INJECTION"


class Deepset(HuggingFace[dict[str, Any], dict[str, Any]]):
    """Wrapper for prompt injection detection model from Deepset.

    For more information, please see the model card:

    - [Deepset](https://huggingface.co/deepset/deberta-v3-base-injection).
    """

    SUPPORTED_MODELS: ClassVar = ["deepset/deberta-v3-base-injection"]

    def _post_processing(
        self, model_outputs: GuardrailInferenceOutput[dict[str, Any]]
    ) -> GuardrailOutput[bool, None, float]:
        return _match_injection_label(model_outputs, DEEPSET_INJECTION_LABEL, self.model.config.id2label)
