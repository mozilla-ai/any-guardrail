from typing import Any, ClassVar

from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, Llama4ForConditionalGeneration

from any_guardrail.base import GuardrailOutput, ThreeStageGuardrail
from any_guardrail.providers.base import StandardProvider
from any_guardrail.providers.huggingface import HuggingFaceProvider
from any_guardrail.types import AnyDict, GuardrailInferenceOutput, GuardrailPreprocessOutput

LlamaGuardPreprocessData = AnyDict  # {"messages": list, "chat_template_kwargs": dict}
LlamaGuardInferenceData = AnyDict  # {"generated_text": str, ...} (shape from provider.generate_chat)


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
        self.model_id = model_id or self.SUPPORTED_MODELS[0]
        if self._is_version_4:
            # v4 wants the standard "add the assistant prefix" template behavior;
            # provider.generate_chat already defaults to that, so no override.
            self._chat_template_kwargs: AnyDict = {}
            default_provider = HuggingFaceProvider(
                model_class=Llama4ForConditionalGeneration,
                tokenizer_class=AutoProcessor,
            )
        elif self.model_id in self.SUPPORTED_MODELS:
            # Llama Guard 3 expects to evaluate the conversation as-is, without an
            # appended assistant prefix.
            self._chat_template_kwargs = {"add_generation_prompt": False}
            default_provider = HuggingFaceProvider(
                model_class=AutoModelForCausalLM,
                tokenizer_class=AutoTokenizer,
            )
        else:
            msg = f"Unsupported model_id: {self.model_id}"
            raise ValueError(msg)

        self.provider = provider or default_provider
        self.provider.load_model(self.model_id)

    def _build_conversation(self, input_text: str, output_text: str | None) -> list[AnyDict]:
        """Shape the chat conversation per model variant."""
        uses_multimodal_content = self.model_id == self.SUPPORTED_MODELS[0] or self._is_version_4
        if uses_multimodal_content:
            user_turn: AnyDict = {"role": "user", "content": [{"type": "text", "text": input_text}]}
        else:
            user_turn = {"role": "user", "content": input_text}
        conversation: list[AnyDict] = [user_turn]
        if output_text:
            if uses_multimodal_content:
                conversation.append({"role": "assistant", "content": [{"type": "text", "text": output_text}]})
            else:
                conversation.append({"role": "assistant", "content": output_text})
        return conversation

    def _pre_processing(
        self, input_text: str, output_text: str | None = None, **kwargs: Any
    ) -> GuardrailPreprocessOutput[LlamaGuardPreprocessData]:
        conversation = self._build_conversation(input_text, output_text)
        chat_template_kwargs: AnyDict = {**self._chat_template_kwargs, **kwargs}
        return GuardrailPreprocessOutput(
            data={
                "messages": conversation,
                "chat_template_kwargs": chat_template_kwargs,
            }
        )

    def _inference(
        self, model_inputs: GuardrailPreprocessOutput[LlamaGuardPreprocessData]
    ) -> GuardrailInferenceOutput[LlamaGuardInferenceData]:
        """Dispatch to ``provider.generate_chat`` with version-appropriate gen params."""
        max_new_tokens = 10 if self._is_version_4 else 20
        # Llama Guard 3 was historically generated with ``pad_token_id=0``; preserve
        # that to keep generation behavior bit-identical with the pre-refactor path.
        generation_kwargs: AnyDict | None = None if self._is_version_4 else {"pad_token_id": 0}
        return self.provider.generate_chat(
            messages=model_inputs.data["messages"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            chat_template_kwargs=model_inputs.data["chat_template_kwargs"] or None,
            generation_kwargs=generation_kwargs,
        )

    def _post_processing(
        self, model_outputs: GuardrailInferenceOutput[LlamaGuardInferenceData]
    ) -> GuardrailOutput[bool, str, None]:
        explanation: str = model_outputs.data["generated_text"]
        if "unsafe" in explanation.lower():
            return GuardrailOutput(valid=False, explanation=explanation)
        return GuardrailOutput(valid=True, explanation=explanation)

    @property
    def _is_version_4(self) -> bool:
        return self.model_id == self.SUPPORTED_MODELS[-1]
