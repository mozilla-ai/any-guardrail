from typing import Any, ClassVar

from any_guardrail.base import GuardrailOutput, ThreeStageGuardrail
from any_guardrail.guardrails.utils import match_injection_label
from any_guardrail.providers.huggingface import HuggingFaceProvider
from any_guardrail.types import GuardrailInferenceOutput, GuardrailPreprocessOutput

INJECGUARD_LABEL = "injection"


class InjecGuard(ThreeStageGuardrail[dict[str, Any], dict[str, Any], bool, None, float]):
    """Prompt injection detection encoder based model.

    For more information, please see the model card:

    - [InjecGuard](https://huggingface.co/leolee99/InjecGuard).
    """

    SUPPORTED_MODELS: ClassVar = ["leolee99/InjecGuard"]

    def __init__(self, model_id: str | None = None, provider: HuggingFaceProvider | None = None) -> None:
        """Initialize the InjecGuard guardrail."""
        self.model_id = model_id or self.SUPPORTED_MODELS[0]
        if self.model_id not in self.SUPPORTED_MODELS:
            msg = f"Only supports {self.SUPPORTED_MODELS}. Please use this path to instantiate model."
            raise ValueError(msg)
        self.provider = provider or HuggingFaceProvider(trust_remote_code=True)
        self.provider.load_model(self.model_id)

    def validate(self, input_text: str) -> GuardrailOutput[bool, None, float]:
        """Validate whether the input text is safe or not."""
        model_inputs = self._pre_processing(input_text)
        model_outputs = self._inference(model_inputs)
        return self._post_processing(model_outputs)

    def _pre_processing(self, input_text: str) -> GuardrailPreprocessOutput[dict[str, Any]]:
        return self.provider.pre_process(input_text)

    def _inference(
        self, model_inputs: GuardrailPreprocessOutput[dict[str, Any]]
    ) -> GuardrailInferenceOutput[dict[str, Any]]:
        return self.provider.infer(model_inputs)

    def _post_processing(
        self, model_outputs: GuardrailInferenceOutput[dict[str, Any]]
    ) -> GuardrailOutput[bool, None, float]:
        return match_injection_label(model_outputs, INJECGUARD_LABEL, self.provider.model.config.id2label)
