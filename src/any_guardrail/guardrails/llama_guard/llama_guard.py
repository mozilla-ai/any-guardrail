from typing import Any, ClassVar

from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, Llama4ForConditionalGeneration

from any_guardrail.base import GuardrailOutput
from any_guardrail.guardrails.huggingface import HuggingFace


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
        "hf-internal-testing/tiny-random-LlamaForCausalLM",
        "meta-llama/Llama-Guard-4-12B",
    ]

    def __init__(self, model_id: str | None = None) -> None:
        """Llama guard model. Either Llama Guard 3 or 4 depending on the model id. Defaults to Llama Guard 3."""
        self.model_id = model_id or self.SUPPORTED_MODELS[0]
        if self._is_version_4:
            self.tokenizer_class = AutoProcessor
            self.model_class = Llama4ForConditionalGeneration
            self.tokenizer_params = {
                "return_tensors": "pt",
                "add_generation_prompt": True,
                "tokenize": True,
                "return_dict": True,
            }
        elif self.model_id in self.SUPPORTED_MODELS:
            self.tokenizer_class = AutoTokenizer  # type: ignore[assignment]
            self.model_class = AutoModelForCausalLM  # type: ignore[assignment]
            self.tokenizer_params = {
                "return_tensors": "pt",
            }
        else:
            msg = f"Unsupported model_id: {self.model_id}"
            raise ValueError(msg)
        super().__init__(model_id)

    def validate(self, input_text: str, output_text: str | None = None, **kwargs: Any) -> GuardrailOutput:
        """Judge whether the input text or the input text, output text pair are unsafe based on the Llama taxonomy.

        Args:
            input_text: the prior text before hitting a system or model.
            output_text: the succeeding text after hitting a system or model.
            **kwargs: additional keyword arguments, specifically supporting 'excluded_category_keys' and 'categories'.
                Please see Llama Guard documentation for more details.

        Returns:
            Provides an explanation that can be parsed to see whether the text is safe or not.

        """
        model_inputs = self._pre_processing(input_text, output_text, **kwargs)
        model_outputs = self._inference(model_inputs)
        return self._post_processing(model_outputs)

    def _load_model(self) -> None:
        self.tokenizer = self.tokenizer_class.from_pretrained(self.model_id)  # type: ignore[no-untyped-call]
        self.model = self.model_class.from_pretrained(self.model_id)

    def _pre_processing(self, input_text: str, output_text: str | None = None, **kwargs: Any) -> Any:
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
        self.model_inputs = self.tokenizer.apply_chat_template(conversation, **self.tokenizer_params, **kwargs)
        return self.model_inputs

    def _inference(self, model_inputs: Any) -> Any:
        if self._is_version_4:
            return self.model.generate(**model_inputs, max_new_tokens=10, do_sample=False)
        return self.model.generate(
            model_inputs,
            max_new_tokens=20,
            pad_token_id=0,
        )

    def _post_processing(self, model_outputs: Any) -> GuardrailOutput:
        if self._is_version_4:
            explanation = self.tokenizer.batch_decode(
                model_outputs[:, self.model_inputs["input_ids"].shape[-1] :], skip_special_tokens=True
            )[0]

            if "unsafe" in explanation.lower():
                return GuardrailOutput(valid=False, explanation=explanation)
            return GuardrailOutput(valid=True, explanation=explanation)

        prompt_len = self.model_inputs.shape[1]
        output = model_outputs[:, prompt_len:]
        explanation = self.tokenizer.decode(output[0])

        if "unsafe" in explanation.lower():
            return GuardrailOutput(valid=False, explanation=explanation)
        return GuardrailOutput(valid=True, explanation=explanation)

    @property
    def _is_version_4(self) -> bool:
        return self.model_id == self.SUPPORTED_MODELS[-1]
