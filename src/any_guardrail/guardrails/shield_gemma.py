from any_guardrail.guardrails.guardrail import Guardrail
from any_guardrail.types import ClassificationOutput, GuardrailModel
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel
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

    def __init__(self, model_id: str, policy: str, threshold: float = DEFAULT_THRESHOLD) -> None:
        super().__init__(model_id)
        supported_models = [
            "google/shieldgemma-2b",
            "google/shieldgemma-9b",
            "google/shieldgemma-27b",
            "hf-internal-testing/tiny-random-Gemma3ForCausalLM",
        ]
        if self.model_id in supported_models:
            self.guardrail = self._model_instantiation()
        else:
            raise ValueError(
                f"Must use one of the following keyword arguments to instantiate model: \n\n {supported_models}"
            )
        self.policy = policy
        self.system_prompt = SYSTEM_PROMPT_SHIELD_GEMMA
        self.threshold = threshold

    def safety_review(self, input_text: str) -> ClassificationOutput:
        """
        Classify input_text according to the safety policy.

        Args:
            input_text: the text you want to safety_review based on the policy
        Returns:
            True if the text violates the policy, False otherwise
        """
        formatted_prompt = self.system_prompt.format(user_prompt=input_text, safety_policy=self.policy)
        if self.guardrail.tokenizer:
            if isinstance(self.guardrail.model, PreTrainedModel):
                inputs = self.guardrail.tokenizer(formatted_prompt, return_tensors="pt").to(self.guardrail.model.device)
                with torch.no_grad():
                    logits = self.guardrail.model(**inputs).logits
            else:
                raise TypeError("Using wrong model type to instantiate Shield Gemma models.")
            vocab = self.guardrail.tokenizer.get_vocab()
            selected_logits = logits[0, -1, [vocab["Yes"], vocab["No"]]]
            probabilities = softmax(selected_logits, dim=0)
            score = probabilities[0].item()

            return ClassificationOutput(unsafe=score > self.threshold)
        else:
            raise TypeError("Did not instantiate tokenizer.")

    def _model_instantiation(self) -> GuardrailModel:
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)  # type: ignore[no-untyped-call]
        model = AutoModelForCausalLM.from_pretrained(self.model_id, device_map="auto", torch_dtype=torch.bfloat16)
        return GuardrailModel(model=model, tokenizer=tokenizer)
