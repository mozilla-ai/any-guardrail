from typing import Any, ClassVar

from any_guardrail.base import GuardrailOutput
from any_guardrail.guardrails.huggingface import HuggingFace, _match_injection_label

JASPER_INJECTION_LABEL = "INJECTION"


class Jasper(HuggingFace):
    """Prompt injection detection encoder based models.

    For more information, please see the model card:

    - [Jasper Deberta](https://huggingface.co/JasperLS/deberta-v3-base-injection)
    - [Jasper Gelectra](https://huggingface.co/JasperLS/gelectra-base-injection).

    Args:
        model_id: HuggingFace path to model.

    Raises:
        ValueError: Can only use model paths for Jasper models from HuggingFace.

    """

    SUPPORTED_MODELS: ClassVar = ["JasperLS/gelectra-base-injection", "JasperLS/deberta-v3-base-injection"]

    def _post_processing(self, model_outputs: dict[str, Any]) -> GuardrailOutput:
        return _match_injection_label(model_outputs, JASPER_INJECTION_LABEL, self.model.config.id2label)
