from typing import Any, ClassVar

from any_guardrail.base import StandardGuardrail
from any_guardrail.guardrails.utils import default, match_injection_label, match_injection_label_batch
from any_guardrail.providers.base import StandardProvider
from any_guardrail.providers.huggingface import HuggingFaceProvider
from any_guardrail.types import GuardrailOutput, StandardInferenceOutput, StandardPreprocessOutput

DEEPSET_INJECTION_LABEL = "INJECTION"


class Deepset(StandardGuardrail):
    """Wrapper for prompt injection detection model from Deepset.

    For more information, please see the model card:

    - [Deepset](https://huggingface.co/deepset/deberta-v3-base-injection).
    """

    SUPPORTED_MODELS: ClassVar = ["deepset/deberta-v3-base-injection"]

    def __init__(self, model_id: str | None = None, provider: StandardProvider | None = None) -> None:
        """Initialize the Deepset guardrail."""
        self.model_id = default(model_id, self.SUPPORTED_MODELS)
        self.provider = provider or HuggingFaceProvider()
        self.provider.load_model(self.model_id)

    def _pre_processing(self, input_text: str) -> StandardPreprocessOutput:
        return self.provider.pre_process(input_text)

    def _inference(self, model_inputs: StandardPreprocessOutput) -> StandardInferenceOutput:
        return self.provider.infer(model_inputs)

    def _post_processing(self, model_outputs: StandardInferenceOutput) -> GuardrailOutput:
        return match_injection_label(model_outputs, DEEPSET_INJECTION_LABEL)

    def _validate_batch(self, input_texts: list[str], **kwargs: Any) -> list[GuardrailOutput]:
        model_inputs = self.provider.pre_process(input_texts, **kwargs)
        model_outputs = self.provider.infer(model_inputs)
        return match_injection_label_batch(model_outputs, DEEPSET_INJECTION_LABEL)
