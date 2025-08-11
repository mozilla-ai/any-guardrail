from any_guardrail.guardrail import Guardrail
from any_guardrail.types import GuardrailOutput
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.nn.functional import softmax

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


class ShieldGemma(Guardrail):
    """
    Wrapper class for Google ShieldGemma models. For more information, please visit the model cards:
    [Shield Gemma](https://huggingface.co/collections/google/shieldgemma-67d130ef8da6af884072a789)

    Note we do not support the image classifier.

    Args:
        model_id: HuggingFace path to model.
        policy: The safety policy to enforce.

    Raises:
        ValueError: Can only use model_ids to ShieldGemma from HuggingFace.
    """

    SUPPORTED_MODELS = [
        "google/shieldgemma-2b",
        "google/shieldgemma-9b",
        "google/shieldgemma-27b",
        "hf-internal-testing/tiny-random-Gemma3ForCausalLM",
    ]

    def __init__(self, model_id: str, policy: str, threshold: float = DEFAULT_THRESHOLD) -> None:
        super().__init__(model_id)
        self.policy = policy
        self.system_prompt = SYSTEM_PROMPT_SHIELD_GEMMA
        self.threshold = threshold

    def validate(self, input_text: str) -> GuardrailOutput:
        """
        Classify input_text according to the safety policy.

        Args:
            input_text: the text you want to validate based on the policy
        Returns:
            True if the text violates the policy, False otherwise
        """
        preprocessed_input = self._pre_processing(input_text)
        logits = self._inference(preprocessed_input)
        unsafe = self._post_processing(logits)
        return GuardrailOutput(unsafe=unsafe)

    def _load_model(self) -> None:
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)  # type: ignore[no-untyped-call]
        model = AutoModelForCausalLM.from_pretrained(self.model_id, device_map="auto", torch_dtype=torch.bfloat16)
        self.model = model
        self.tokenizer = tokenizer

    def _pre_processing(self, input_text: str) -> torch.Tensor:
        formatted_prompt = self.system_prompt.format(user_prompt=input_text, safety_policy=self.policy)
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
        return inputs

    def _inference(self, inputs: torch.Tensor) -> torch.FloatTensor:
        with torch.no_grad():
            logits = self.model(**inputs).logits
        return logits

    def _post_processing(self, logits: torch.FloatTensor) -> bool:
        vocab = self.tokenizer.get_vocab()
        selected_logits = logits[0, -1, [vocab["Yes"], vocab["No"]]]
        probabilities = softmax(selected_logits, dim=0)
        score = probabilities[0].item()
        return score > self.threshold
