from any_guardrail.guardrails.guardrail import Guardrail
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.nn.functional import softmax
from typing import Any

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
    https://huggingface.co/collections/google/shieldgemma-67d130ef8da6af884072a789

    Note we do not support the image classifier.

    Args:
        modelpath (str): HuggingFace path to model.
        policy (str): The safety policy to enforce.
    """

    def __init__(self, modelpath: str, policy: str, threshold: float = DEFAULT_THRESHOLD) -> None:
        super().__init__(modelpath)
        if self.modelpath in ["google/shieldgemma-2b", "google/shieldgemma-9b", "google/shieldgemma-27b"]:
            self.model, self.tokenizer = self._model_instantiation()
        else:
            raise ValueError(
                "Must use one of the following keyword arguments to instantiate model: "
                "\n\n google/shieldgemma-2b \n google/shieldgemma-9b \n google/shieldgemma-27b"
            )
        self.policy = policy
        self.system_prompt = SYSTEM_PROMPT_SHIELD_GEMMA
        self.threshold = threshold

    def classify(self, input_text: str) -> bool:
        """
        Classify input_text according to the safety policy.

        Args:
            input_text: the text you want to classify based on the policy
        Returns:
            True if the text violates the policy, False otherwise
        """
        self.system_prompt.format(user_prompt=input_text, safety_policy=self.policy)
        inputs = self.tokenizer(self.system_prompt, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**inputs).logits
        vocab = self.tokenizer.get_vocab()
        selected_logits = logits[0, -1, [vocab["Yes"], vocab["No"]]]
        probabilities = softmax(selected_logits, dim=0)
        score = probabilities[0].item()

        return score > self.threshold

    def _model_instantiation(self) -> tuple[Any, Any]:
        tokenizer = AutoTokenizer.from_pretrained(self.modelpath)  # type: ignore[no-untyped-call]
        model = AutoModelForCausalLM.from_pretrained(self.modelpath, device_map="auto", torch_dtype=torch.bfloat16)
        return model, tokenizer
