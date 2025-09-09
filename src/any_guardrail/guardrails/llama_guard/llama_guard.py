from typing import Any, ClassVar

from any_guardrail.base import GuardrailOutput
from any_guardrail.guardrails.huggingface import HuggingFace
from any_guardrail.guardrails.llama_guard.llama_guard3 import LlamaGuard3
from any_guardrail.guardrails.llama_guard.llama_guard4 import LlamaGuard4


class LlamaGuard(HuggingFace):
    """Wrapper class for Llama Guard 3 & 4 implementations.

    For more information about the implementations about either off topic model, please see the below model cards:

    - [Meta Llama Guard 3 Docs](https://www.llama.com/docs/model-cards-and-prompt-formats/llama-guard-3/)
    - [HuggingFace Llama Guard 3 Docs](https://huggingface.co/meta-llama/Llama-Guard-3-1B)
    - [Meta Llama Guard 4 Docs](https://www.llama.com/docs/model-cards-and-prompt-formats/llama-guard-4/)
    - [HuggingFace Llama Guard 4 Docs](https://huggingface.co/meta-llama/Llama-Guard-4-12B)
    """

    SUPPORTED_MODELS: ClassVar = [
        "meta-llama/Llama-Guard-3-1B",
        "meta-llama/Llama-Guard-3-8B",
        "meta-llama/Llama-Guard-4-12B",
    ]

    implementation: LlamaGuard3 | LlamaGuard4

    def __init__(self, model_id: str | None = None) -> None:
        """Llama guard model. Either Llama Guard 3 or 4 depending on the model id. Defaults to Llama Guard 3."""
        super().__init__(model_id)
        if (self.model_id == self.SUPPORTED_MODELS[0]) or (self.model_id == self.SUPPORTED_MODELS[1]):
            self.implementation = LlamaGuard3()
        elif self.model_id == self.SUPPORTED_MODELS[2]:
            self.implementation = LlamaGuard4()
        else:
            msg = f"Unsupported model_id: {self.model_id}"
            raise ValueError(msg)
        super().__init__()

    def validate(self, input_text: str, output_text: str | None = None) -> GuardrailOutput:
        """Judge whether the input text or the input text, output text pair are unsafe based on the Llama taxonomy

        Args:
            input_text: the prior text before hitting a system or model.
            output_text: the succeeding text after hitting a system or model.

        Returns:
            Provides an explanation that can be parsed to see whether the text is safe or not.

        """
        model_inputs = self.implementation._pre_processing(input_text, output_text)
        model_outputs = self.implementation._inference(model_inputs)
        return self._post_processing(model_outputs)

    def _load_model(self) -> None:
        self.implementation._load_model()

    def _post_processing(self, model_outputs: Any) -> GuardrailOutput:
        return self.implementation._post_processing(model_outputs)
