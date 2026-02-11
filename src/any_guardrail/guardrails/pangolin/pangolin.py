from typing import ClassVar

from any_guardrail.base import StandardGuardrail
from any_guardrail.guardrails.utils import default, match_injection_label
from any_guardrail.providers.base import StandardProvider
from any_guardrail.providers.huggingface import HuggingFaceProvider
from any_guardrail.types import BinaryScoreOutput, StandardInferenceOutput, StandardPreprocessOutput

PANGOLIN_INJECTION_LABEL = "unsafe"


class Pangolin(StandardGuardrail):
    """Prompt injection detection encoder based models.

    For more information, please see the model card:

    - [Pangolin Base](https://huggingface.co/dcarpintero/pangolin-guard-base)
    """

    SUPPORTED_MODELS: ClassVar = ["dcarpintero/pangolin-guard-base"]

    def __init__(self, model_id: str | None = None, provider: StandardProvider | None = None) -> None:
        """Initialize the Pangolin guardrail."""
        self.model_id = default(model_id, self.SUPPORTED_MODELS)
        self.provider = provider or HuggingFaceProvider()
        self.provider.load_model(self.model_id)

    def _pre_processing(self, input_text: str) -> StandardPreprocessOutput:
        return self.provider.pre_process(input_text)

    def _inference(self, model_inputs: StandardPreprocessOutput) -> StandardInferenceOutput:
        return self.provider.infer(model_inputs)

    def _post_processing(self, model_outputs: StandardInferenceOutput) -> BinaryScoreOutput:
        return match_injection_label(model_outputs, PANGOLIN_INJECTION_LABEL, self.provider.model.config.id2label)  # type: ignore[attr-defined]
