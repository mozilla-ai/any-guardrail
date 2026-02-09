from typing import ClassVar

from any_guardrail.base import StandardGuardrail
from any_guardrail.guardrails.utils import default, match_injection_label
from any_guardrail.providers.base import StandardProvider
from any_guardrail.providers.huggingface import HuggingFaceProvider
from any_guardrail.types import BinaryScoreOutput, StandardInferenceOutput, StandardPreprocessOutput

PROTECTAI_INJECTION_LABEL = "INJECTION"


class Protectai(StandardGuardrail):
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

    def __init__(self, model_id: str | None = None, provider: StandardProvider | None = None) -> None:
        """Initialize the Protectai guardrail."""
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
        return match_injection_label(model_outputs, PROTECTAI_INJECTION_LABEL, self.provider.model.config.id2label)  # type: ignore[attr-defined]
