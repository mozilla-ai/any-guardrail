"""
Shieldgemma guardrail for policy-based safety classification using a language model.
"""
from any_guardrail.guardrails.guardrail import Guardrail
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.nn.functional import softmax
from typing import Any
from ..utils.constants import DEFAULT_THRESHOLD, LABEL_UNSAFE, LABEL_SAFE, LABEL_YES, LABEL_NO, SYSTEM_PROMPT_SHIELD_GEMMA

class Shieldgemma(Guardrail):
    """
    Guardrail for classifying prompts according to a safety policy using a language model.
    Args:
        modelpath (str): Path to the model.
        policy (str): The safety policy to enforce.
    """
    def __init__(self, modelpath: str, policy: str) -> None:
        """
        Initialize Shieldgemma with model path and policy.
        """
        self.modelpath = modelpath
        try:
            self.model, self.tokenizer = self.model_instantiation()
        except Exception as e:
            raise RuntimeError(f"Failed to load model or tokenizer: {e}")
        self.policy = policy
        self.system_prompt = SYSTEM_PROMPT_SHIELD_GEMMA
        self.threshold = DEFAULT_THRESHOLD

    def classify(self, input_text: str) -> str:
        """
        Classify input_text according to the safety policy.
        Returns 'UNSAFE' or 'SAFE'.
        """
        try:
            self.system_prompt.format(user_prompt=input_text, safety_policy=self.policy)
            inputs = self.tokenizer(self.system_prompt, return_tensors="pt")
            with torch.no_grad():
                logits = self.model(**inputs).logits
            vocab = self.tokenizer.get_vocab()
            selected_logits = logits[0, -1, [vocab[LABEL_YES], vocab[LABEL_NO]]]
            probabilities=softmax(selected_logits, dim=0)
            score=probabilities[0].item()
        except Exception as e:
            raise RuntimeError(f"Error during classification: {e}")

        if score > self.threshold:
            return LABEL_UNSAFE
        else:
            return LABEL_SAFE

    def model_instantiation(self) -> tuple[Any, Any]:
        """
        Load the model and tokenizer from the given model path.
        Returns:
            model, tokenizer
        """
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.modelpath)
            model = AutoModelForCausalLM.from_pretrained(self.modelpath, device_map="auto", torch_dtype=torch.bfloat16)
            return model, tokenizer
        except Exception as e:
            raise RuntimeError(f"Error loading model/tokenizer: {e}")
    