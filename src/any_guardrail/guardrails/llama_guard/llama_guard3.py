from typing import Any, ClassVar

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from any_guardrail.base import GuardrailOutput
from any_guardrail.guardrails.huggingface import HuggingFace


class LlamaGuard3(HuggingFace):
    """Wrapper for Llama Guard 3.

    For more information, please view the documentation that Meta provides:

    - [Meta Llama Guard Docs](https://www.llama.com/docs/model-cards-and-prompt-formats/llama-guard-3/)
    -[HuggingFace Llama Guard 3 Docs](https://huggingface.co/meta-llama/Llama-Guard-3-1B)
    """

    SUPPORTED_MODELS: ClassVar = ["meta-llama/Llama-Guard-3-1B",
                                  "meta-llama/Llama-Guard-3-8B",]

    def _load_model(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)  # type: ignore[no-untyped-call]
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=torch.bfloat16)

    def _pre_processing(self, 
                        input_text: str, 
                        output_text: str | None = None,
                        **kwargs) -> Any:
        if output_text:
            conversation = [{"role": "user","content": [{"type": "text", "text": input_text},],},
                            {"role": "assistant","content": [{"type": "text", "text": output_text},],}]
        else:
            conversation = [{"role": "user","content": [{"type": "text", "text": input_text},],},]
        return self.tokenizer.apply_chat_template(conversation, 
                                                  return_tensors="pt",
                                                  **kwargs)

    def _inference(self, model_inputs: Any) -> Any:
        prompt_len = model_inputs.shape[1]
        output = self.model.generate(model_inputs, max_new_tokens=20, pad_token_id=0,)
        return output[:, prompt_len:]

    def _post_processing(self, model_outputs: Any) -> GuardrailOutput:
        explanation = self.tokenizer.decode(model_outputs[0])
        return GuardrailOutput(explanation=explanation)
