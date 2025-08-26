from typing import Any, ClassVar

from any_guardrail.base import GuardrailOutput
from any_guardrail.guardrails.huggingface import HuggingFace, _match_injection_label

INJECGUARD_LABEL = "injection"


class InjecGuard(HuggingFace):
    """Prompt injection detection encoder based model.

    For more information, please see the model card:

    - [InjecGuard](https://huggingface.co/leolee99/InjecGuard).
    """

    SUPPORTED_MODELS: ClassVar = ["leolee99/InjecGuard"]

    def _load_model(self) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)  # type: ignore[no-untyped-call]

    def _post_processing(self, model_outputs: dict[str, Any]) -> GuardrailOutput:
        return _match_injection_label(model_outputs, INJECGUARD_LABEL, self.model.config.id2label)
