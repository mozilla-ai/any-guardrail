from typing import ClassVar

from any_guardrail.base import StandardGuardrail
from any_guardrail.guardrails.utils import default, match_injection_label
from any_guardrail.providers.base import StandardProvider
from any_guardrail.providers.huggingface import HuggingFaceProvider
from any_guardrail.types import BinaryScoreOutput, StandardInferenceOutput, StandardPreprocessOutput

SENTINEL_INJECTION_LABEL = "jailbreak"


class Sentinel(StandardGuardrail):
    """Prompt injection detection encoder based model.

    For more information, please see the model card:

    - [Sentinel](https://huggingface.co/qualifire/prompt-injection-sentinel).
    """

    SUPPORTED_MODELS: ClassVar = ["qualifire/prompt-injection-sentinel"]

    def __init__(self, model_id: str | None = None, provider: StandardProvider | None = None) -> None:
        """Initialize the Sentinel guardrail."""
        self.model_id = default(model_id, self.SUPPORTED_MODELS)
        self.provider = provider or HuggingFaceProvider()
        self.provider.load_model(self.model_id)

    def validate(self, input_text: str) -> BinaryScoreOutput:
        """Validate whether the input text is safe or not."""
        model_inputs = self._pre_processing(input_text)
        model_outputs = self._inference(model_inputs)
        return self._post_processing(model_outputs)

    def _pre_processing(self, input_text: str) -> StandardPreprocessOutput:
        return self.provider.pre_process(input_text)

    def _inference(self, model_inputs: StandardPreprocessOutput) -> StandardInferenceOutput:
        return self.provider.infer(model_inputs)

    def _post_processing(self, model_outputs: StandardInferenceOutput) -> BinaryScoreOutput:
        return match_injection_label(model_outputs, SENTINEL_INJECTION_LABEL, self.provider.model.config.id2label)  # type: ignore[attr-defined]
