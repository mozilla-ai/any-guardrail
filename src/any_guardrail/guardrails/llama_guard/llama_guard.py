from typing import Any, ClassVar

import torch

from any_guardrail.base import GuardrailOutput, ThreeStageGuardrail
from any_guardrail.providers.base import StandardProvider
from any_guardrail.providers.huggingface import HuggingFaceProvider
from any_guardrail.types import AnyDict, GuardrailInferenceOutput, GuardrailPreprocessOutput

# Type alias for LlamaGuard - preprocessing can return either a dict (v4) or tensor (v3)
LlamaGuardPreprocessData = AnyDict | Any  # dict for v4, tensor for v3
LlamaGuardInferenceData = Any  # Generated tensor output


class LlamaGuard(ThreeStageGuardrail[LlamaGuardPreprocessData, LlamaGuardInferenceData, bool, str, None]):
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

    def __init__(
        self,
        model_id: str | None = None,
        provider: StandardProvider | None = None,
    ) -> None:
        """Llama guard model. Either Llama Guard 3 or 4 depending on the model id. Defaults to Llama Guard 3."""
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, Llama4ForConditionalGeneration

        self.model_id = model_id or self.SUPPORTED_MODELS[0]
        if self._is_version_4:
            self.tokenizer_params: AnyDict = {
                "return_tensors": "pt",
                "add_generation_prompt": True,
                "tokenize": True,
                "return_dict": True,
            }
            default_provider = HuggingFaceProvider(
                model_class=Llama4ForConditionalGeneration,
                tokenizer_class=AutoProcessor,
            )
        elif self.model_id in self.SUPPORTED_MODELS:
            self.tokenizer_params = {
                "return_tensors": "pt",
            }
            default_provider = HuggingFaceProvider(
                model_class=AutoModelForCausalLM,
                tokenizer_class=AutoTokenizer,
            )
        else:
            msg = f"Unsupported model_id: {self.model_id}"
            raise ValueError(msg)

        self.provider = provider or default_provider
        self.provider.load_model(self.model_id)

    def _pre_processing(
        self, input_text: str, output_text: str | None = None, **kwargs: Any
    ) -> GuardrailPreprocessOutput[LlamaGuardPreprocessData]:
        if output_text:
            if self.model_id == self.SUPPORTED_MODELS[0] or self._is_version_4:
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": input_text},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": output_text},
                        ],
                    },
                ]
            else:
                conversation = [
                    {
                        "role": "user",
                        "content": input_text,
                    },
                    {
                        "role": "assistant",
                        "content": output_text,
                    },
                ]
        else:
            if self.model_id == self.SUPPORTED_MODELS[0] or self._is_version_4:
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": input_text},
                        ],
                    },
                ]
            else:
                conversation = [
                    {
                        "role": "user",
                        "content": input_text,
                    },
                ]
        self._cached_model_inputs = self.provider.tokenizer.apply_chat_template(  # type: ignore[attr-defined]
            conversation, **self.tokenizer_params, **kwargs
        )
        return GuardrailPreprocessOutput(data=self._cached_model_inputs)

    def _inference(
        self, model_inputs: GuardrailPreprocessOutput[LlamaGuardPreprocessData]
    ) -> GuardrailInferenceOutput[LlamaGuardInferenceData]:
        """Run generate() for inference."""
        with torch.no_grad():
            if self._is_version_4:
                output = self.provider.model.generate(**model_inputs.data, max_new_tokens=10, do_sample=False)  # type: ignore[attr-defined]
            else:
                output = self.provider.model.generate(  # type: ignore[attr-defined]
                    model_inputs.data["input_ids"],
                    max_new_tokens=20,
                    pad_token_id=0,
                )
        return GuardrailInferenceOutput(data=output)

    def _post_processing(
        self, model_outputs: GuardrailInferenceOutput[LlamaGuardInferenceData]
    ) -> GuardrailOutput[bool, str, None]:
        if self._is_version_4:
            explanation = self.provider.tokenizer.batch_decode(  # type: ignore[attr-defined]
                model_outputs.data[:, self._cached_model_inputs["input_ids"].shape[-1] :],
                skip_special_tokens=True,
            )[0]

            if "unsafe" in explanation.lower():
                return GuardrailOutput(valid=False, explanation=explanation)
            return GuardrailOutput(valid=True, explanation=explanation)

        prompt_len = self._cached_model_inputs.get("input_ids").shape[1]
        output = model_outputs.data[:, prompt_len:]
        explanation = self.provider.tokenizer.decode(output[0])  # type: ignore[attr-defined]

        if "unsafe" in explanation.lower():
            return GuardrailOutput(valid=False, explanation=explanation)
        return GuardrailOutput(valid=True, explanation=explanation)

    @property
    def _is_version_4(self) -> bool:
        return self.model_id == self.SUPPORTED_MODELS[-1]
