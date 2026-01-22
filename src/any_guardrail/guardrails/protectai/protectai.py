from typing import Any, ClassVar

from any_guardrail.base import GuardrailOutput
from any_guardrail.guardrails.huggingface import HuggingFace, _match_injection_label
from any_guardrail.types import GuardrailInferenceOutput

PROTECTAI_INJECTION_LABEL = "INJECTION"


class Protectai(HuggingFace[dict[str, Any], dict[str, Any], bool, None, float]):
    """Prompt injection detection encoder based models.

    For more information, please see the model card:

    - [ProtectAI](https://huggingface.co/collections/protectai/llm-security-65c1f17a11c4251eeab53f40).
    """

    SUPPORTED_MODELS: ClassVar = [
        "ProtectAI/deberta-v3-small-prompt-injection-v2",
        "ProtectAI/distilroberta-base-rejection-v1",
        "ProtectAI/deberta-v3-base-prompt-injection",
        "ProtectAI/deberta-v3-base-prompt-injection-v2",
    ]

    def _post_processing(
        self, model_outputs: GuardrailInferenceOutput[dict[str, Any]]
    ) -> GuardrailOutput[bool, None, float]:
        return _match_injection_label(model_outputs, PROTECTAI_INJECTION_LABEL, self.model.config.id2label)
