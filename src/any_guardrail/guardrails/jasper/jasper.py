from typing import Any, ClassVar

from any_guardrail.base import GuardrailOutput, ThreeStageGuardrail
from any_guardrail.guardrails.utils import match_injection_label
from any_guardrail.providers.huggingface import HuggingFaceProvider
from any_guardrail.types import GuardrailInferenceOutput, GuardrailPreprocessOutput

JASPER_INJECTION_LABEL = "INJECTION"


class Jasper(ThreeStageGuardrail[dict[str, Any], dict[str, Any], bool, None, float]):
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

    def __init__(self, model_id: str | None = None, provider: HuggingFaceProvider | None = None) -> None:
        """Initialize the Jasper guardrail."""
        self.model_id = model_id or self.SUPPORTED_MODELS[0]
        if self.model_id not in self.SUPPORTED_MODELS:
            msg = f"Only supports {self.SUPPORTED_MODELS}. Please use this path to instantiate model."
            raise ValueError(msg)
        self.provider = provider or HuggingFaceProvider()
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
        return match_injection_label(model_outputs, JASPER_INJECTION_LABEL, self.provider.model.config.id2label)
