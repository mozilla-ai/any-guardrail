from typing import Any, ClassVar

from any_guardrail.base import GuardrailOutput, ThreeStageGuardrail
from any_guardrail.providers.huggingface import HuggingFaceProvider
from any_guardrail.types import GuardrailInferenceOutput, GuardrailPreprocessOutput

SYSTEM_PROMPT_SHIELD_GEMMA = """You are a policy expert trying to help determine whether a user
prompt is in violation of the defined safety policies.

<start_of_turn>
Human Question: {user_prompt}
<end_of_turn>

Our safety principle is defined in the below:

{safety_policy}

Does the human question violate the above principle? Your answer must start
with 'Yes' or 'No'. And then walk through step by step to be sure we answer
correctly.
"""
DEFAULT_THRESHOLD: float = 0.5


class ShieldGemma(ThreeStageGuardrail[dict[str, Any], dict[str, Any], bool, None, float]):
    """Wrapper class for Google ShieldGemma models.

    For more information, please visit the model cards: [Shield Gemma](https://huggingface.co/collections/google/shieldgemma-67d130ef8da6af884072a789).

    Note we do not support the image classifier.
    """

    SUPPORTED_MODELS: ClassVar = [
        "google/shieldgemma-2b",
        "google/shieldgemma-9b",
        "google/shieldgemma-27b",
        "hf-internal-testing/tiny-random-Gemma3ForCausalLM",
    ]

    def __init__(
        self,
        policy: str,
        threshold: float = DEFAULT_THRESHOLD,
        model_id: str | None = None,
        provider: HuggingFaceProvider | None = None,
    ) -> None:
        """Initialize the ShieldGemma guardrail."""
        self.model_id = model_id or self.SUPPORTED_MODELS[0]
        if self.model_id not in self.SUPPORTED_MODELS:
            msg = f"Only supports {self.SUPPORTED_MODELS}. Please use this path to instantiate model."
            raise ValueError(msg)
        self.policy = policy
        self.system_prompt = SYSTEM_PROMPT_SHIELD_GEMMA
        self.threshold = threshold
        if provider is not None:
            self.provider = provider
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.provider = HuggingFaceProvider(model_class=AutoModelForCausalLM, tokenizer_class=AutoTokenizer)
        self.provider.load_model(self.model_id)

    def validate(self, input_text: str) -> GuardrailOutput[bool, None, float]:
        """Validate whether the input text is safe or not."""
        model_inputs = self._pre_processing(input_text)
        model_outputs = self._inference(model_inputs)
        return self._post_processing(model_outputs)

    def _pre_processing(self, input_text: str) -> GuardrailPreprocessOutput[dict[str, Any]]:
        formatted_prompt = self.system_prompt.format(user_prompt=input_text, safety_policy=self.policy)
        tokenized = self.provider.tokenizer(formatted_prompt, return_tensors="pt")
        return GuardrailPreprocessOutput(data=tokenized)

    def _inference(
        self, model_inputs: GuardrailPreprocessOutput[dict[str, Any]]
    ) -> GuardrailInferenceOutput[dict[str, Any]]:
        return self.provider.infer(model_inputs)

    def _post_processing(
        self, model_outputs: GuardrailInferenceOutput[dict[str, Any]]
    ) -> GuardrailOutput[bool, None, float]:
        from torch.nn.functional import softmax

        logits = model_outputs.data["logits"]
        vocab = self.provider.tokenizer.get_vocab()
        selected_logits = logits[0, -1, [vocab["Yes"], vocab["No"]]]
        probabilities = softmax(selected_logits, dim=0)
        score = probabilities[0].item()
        return GuardrailOutput(valid=score < self.threshold, explanation=None, score=score)
