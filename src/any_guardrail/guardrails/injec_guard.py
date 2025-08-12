from typing import Any, ClassVar

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from any_guardrail.guardrails.huggingface import HuggingFace, _match_injection_label
from any_guardrail.types import GuardrailOutput

INJECGUARD_LABEL = "injection"


class InjecGuard(HuggingFace):
    """Prompt injection detection encoder based model.

    For more information, please see the model card:

    - [InjecGuard](https://huggingface.co/leolee99/InjecGuard).
    """

    SUPPORTED_MODELS: ClassVar = ["leolee99/InjecGuard"]

    def _load_model(self) -> None:
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

    def _post_processing(self, model_outputs: dict[str, Any]) -> GuardrailOutput:
        return _match_injection_label(model_outputs, INJECGUARD_LABEL, self.model.config.id2label)
