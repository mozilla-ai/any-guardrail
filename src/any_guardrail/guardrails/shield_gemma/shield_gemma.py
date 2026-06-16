from typing import ClassVar

from any_guardrail.base import GuardrailOutput, StandardGuardrail
from any_guardrail.guardrails.utils import default
from any_guardrail.providers.base import StandardProvider
from any_guardrail.providers.huggingface import HuggingFaceProvider
from any_guardrail.types import (
    AnyDict,
    GuardrailPreprocessOutput,
    StandardInferenceOutput,
    StandardPreprocessOutput,
)

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


class ShieldGemma(StandardGuardrail):
    """Wrapper class for Google ShieldGemma models.

    For more information, please visit the model cards: [Shield Gemma](https://huggingface.co/collections/google/shieldgemma-67d130ef8da6af884072a789).

    Note we do not support the image classifier.
    """

    SUPPORTED_MODELS: ClassVar = [
        "google/shieldgemma-2b",
        "google/shieldgemma-9b",
        "google/shieldgemma-27b",
    ]

    def __init__(
        self,
        policy: str,
        threshold: float = DEFAULT_THRESHOLD,
        model_id: str | None = None,
        provider: StandardProvider | None = None,
    ) -> None:
        """Initialize the ShieldGemma guardrail."""
        self.model_id = default(model_id, self.SUPPORTED_MODELS)
        self.policy = policy
        self.system_prompt = SYSTEM_PROMPT_SHIELD_GEMMA
        self.threshold = threshold
        # Lazy-import transformers so importing ShieldGemma does not require
        # the huggingface extra. A caller could supply a non-HF provider that
        # returns torch tensors from infer() — the transformers classes are
        # only needed when the provider is a HuggingFaceProvider (or when we
        # construct one ourselves).
        load_kwargs: AnyDict = {}
        if provider is not None:
            self.provider = provider
            if isinstance(self.provider, HuggingFaceProvider):
                # ShieldGemma is a causal LM. A default-constructed
                # HuggingFaceProvider targets AutoModelForSequenceClassification,
                # which would load Gemma2ForSequenceClassification — the
                # checkpoint has no classification head, so score.weight is
                # randomly initialized and _post_processing later crashes on
                # 2D logits. Enforce the right classes for this load.
                from transformers import AutoModelForCausalLM, AutoTokenizer

                load_kwargs = {"model_class": AutoModelForCausalLM, "tokenizer_class": AutoTokenizer}
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.provider = HuggingFaceProvider(model_class=AutoModelForCausalLM, tokenizer_class=AutoTokenizer)
        self.provider.load_model(self.model_id, **load_kwargs)

    def _pre_processing(self, input_text: str) -> StandardPreprocessOutput:
        formatted_prompt = self.system_prompt.format(user_prompt=input_text, safety_policy=self.policy)
        tokenized = self.provider.tokenizer(formatted_prompt, return_tensors="pt")  # type: ignore[attr-defined]
        device = getattr(self.provider, "device", None)
        if device is not None:
            tokenized = tokenized.to(device)
        return GuardrailPreprocessOutput(data=tokenized)

    def _inference(self, model_inputs: StandardPreprocessOutput) -> StandardInferenceOutput:
        return self.provider.infer(model_inputs)

    def _post_processing(self, model_outputs: StandardInferenceOutput) -> GuardrailOutput:
        # Lazy-import torch so importing ShieldGemma does not require the
        # huggingface extra at module load. ShieldGemma still needs torch
        # at validate() time to slice the causal-LM logits, but a user who
        # never calls validate() can import the class freely.
        from torch.nn.functional import softmax

        logits = model_outputs.data["logits"]
        vocab = self.provider.tokenizer.get_vocab()  # type: ignore[attr-defined]
        selected_logits = logits[0, -1, [vocab["Yes"], vocab["No"]]]
        probabilities = softmax(selected_logits, dim=0)
        score = probabilities[0].item()
        return GuardrailOutput(valid=score < self.threshold, score=score)
