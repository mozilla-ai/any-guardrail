from typing import Any, ClassVar

import torch
from transformers import Llama4ForConditionalGeneration, AutoProcessor

from any_guardrail.base import GuardrailOutput
from any_guardrail.guardrails.huggingface import HuggingFace


class LlamaGuard4(HuggingFace):
    """Wrapper for Llama Guard 4.

    For more information, please view the documentation that Meta provides:

    - [Meta Llama Guard Docs](https://www.llama.com/docs/model-cards-and-prompt-formats/llama-guard-4/)
    - [HuggingFace Llama Guard 4 Docs](https://huggingface.co/meta-llama/Llama-Guard-4-12B)
    """

    SUPPORTED_MODELS: ClassVar = ["meta-llama/Llama-Guard-4-12B"]

    def _load_model(self) -> None:
        self.tokenizer = AutoProcessor.from_pretrained(self.model_id)  # type: ignore[no-untyped-call]
        self.model = Llama4ForConditionalGeneration.from_pretrained(self.model_id, dtype=torch.bfloat16)

    def _pre_processing(self, 
                        input_text: str, 
                        output_text: str | None = None,
                        **kwargs) -> Any:
        if output_text:
            conversation = [{"role": "user","content": [{"type": "text", "text": input_text},],},
                            {"role": "assistant","content": [{"type": "text", "text": output_text},],}]
        else:
            conversation = [{"role": "user","content": [{"type": "text", "text": input_text},],},]
        self.model_inputs =  self.tokenizer.apply_chat_template(
                                        conversation,
                                        tokenize=True,
                                        add_generation_prompt=True,
                                        return_tensors="pt",
                                        return_dict=True,
                                        **kwargs
                                    )
        return self.model_inputs

    def _inference(self, model_inputs: Any) -> Any:
        return self.model.generate(**model_inputs, max_new_tokens=10, do_sample=False), model_inputs

    def _post_processing(self, model_outputs: Any) -> GuardrailOutput:
        explanation = self.tokenizer.batch_decode(model_outputs[:, self.model_inputs["input_ids"].shape[-1]:], skip_special_tokens=True)[0]
        return GuardrailOutput(explanation=explanation)
