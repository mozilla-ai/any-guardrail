from typing import Any, ClassVar

from any_guardrail.base import GuardrailOutput
from any_guardrail.guardrails.huggingface import HuggingFace

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


class ShieldGemma(HuggingFace):
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

    def __init__(self, policy: str, threshold: float = DEFAULT_THRESHOLD, model_id: str | None = None) -> None:
        """Initialize the ShieldGemma guardrail."""
        super().__init__(model_id)
        self.policy = policy
        self.system_prompt = SYSTEM_PROMPT_SHIELD_GEMMA
        self.threshold = threshold

    def _load_model(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)  # type: ignore[no-untyped-call]

    def _pre_processing(self, input_text: str) -> Any:
        formatted_prompt = self.system_prompt.format(user_prompt=input_text, safety_policy=self.policy)
        return super()._pre_processing(formatted_prompt)

    def _post_processing(self, model_outputs: dict[str, Any]) -> GuardrailOutput:
        from torch.nn.functional import softmax

        logits = model_outputs["logits"]
        vocab = self.tokenizer.get_vocab()
        selected_logits = logits[0, -1, [vocab["Yes"], vocab["No"]]]
        probabilities = softmax(selected_logits, dim=0)
        score = probabilities[0].item()
        return GuardrailOutput(valid=score < self.threshold, explanation=None, score=score)
