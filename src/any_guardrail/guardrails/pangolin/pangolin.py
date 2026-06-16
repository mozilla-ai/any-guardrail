from typing import Any, ClassVar

from any_guardrail.base import StandardGuardrail
from any_guardrail.guardrails.utils import default, match_injection_label, match_injection_label_batch
from any_guardrail.providers.base import StandardProvider
from any_guardrail.providers.huggingface import HuggingFaceProvider
from any_guardrail.types import GuardrailOutput, StandardInferenceOutput, StandardPreprocessOutput

PANGOLIN_INJECTION_LABEL = "unsafe"


class Pangolin(StandardGuardrail):
    """Prompt injection detection encoder based models.

    For more information, please see the model cards:

    - [Pangolin Base](https://huggingface.co/dcarpintero/pangolin-guard-base) (default) — ModernBERT-base.
    - [Pangolin Large](https://huggingface.co/dcarpintero/pangolin-guard-large) — ModernBERT-large;
      higher accuracy, 8192-token context. Same ``"unsafe"`` label, so it is a drop-in alternative.
    """

    SUPPORTED_MODELS: ClassVar = ["dcarpintero/pangolin-guard-base", "dcarpintero/pangolin-guard-large"]

    def __init__(self, model_id: str | None = None, provider: StandardProvider | None = None) -> None:
        """Initialize the Pangolin guardrail."""
        self.model_id = default(model_id, self.SUPPORTED_MODELS)
        self.provider = provider or HuggingFaceProvider()
        self.provider.load_model(self.model_id)

    def _pre_processing(self, input_text: str) -> StandardPreprocessOutput:
        return self.provider.pre_process(input_text)

    def _inference(self, model_inputs: StandardPreprocessOutput) -> StandardInferenceOutput:
        return self.provider.infer(model_inputs)

    def _post_processing(self, model_outputs: StandardInferenceOutput) -> GuardrailOutput:
        return match_injection_label(model_outputs, PANGOLIN_INJECTION_LABEL)

    def _validate_batch(self, input_texts: list[str], **kwargs: Any) -> list[GuardrailOutput]:
        model_inputs = self.provider.pre_process(input_texts, **kwargs)
        model_outputs = self.provider.infer(model_inputs)
        return match_injection_label_batch(model_outputs, PANGOLIN_INJECTION_LABEL)
