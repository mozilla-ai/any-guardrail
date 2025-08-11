from typing import Any, Tuple, Dict, List

from any_guardrail.guardrail import Guardrail
from any_guardrail.types import GuardrailOutput
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

DUOGUARD_CATEGORIES = [
    "Violent crimes",
    "Non-violent crimes",
    "Sex-related crimes",
    "Child sexual exploitation",
    "Specialized advice",
    "Privacy",
    "Intellectual property",
    "Indiscriminate weapons",
    "Hate",
    "Suicide and self-harm",
    "Sexual content",
    "Jailbreak prompts",
]

DUOGUARD_DEFAULT_THRESHOLD = 0.5  # Taken from the DuoGuard model card.


class DuoGuard(Guardrail):
    """
    Guardrail that classifies text based on the categories in DUOGUARD_CATEGORIES. For more information, please see the
    model cards: [DuoGuard](https://huggingface.co/collections/DuoGuard/duoguard-models-67a29ad8bd579a404e504d21)

    Args:
        model_id: HuggingFace path to model.

    Raises:
        ValueError: Only supports DuoGuard models from HuggingFace.
    """

    SUPPORTED_MODELS = [
        "DuoGuard/DuoGuard-0.5B",
        "DuoGuard/DuoGuard-1B-Llama-3.2-transfer",
        "DuoGuard/DuoGuard-1.5B-transfer",
    ]

    def __init__(self, model_id: str, threshold: float = DUOGUARD_DEFAULT_THRESHOLD) -> None:
        super().__init__(model_id)
        self.threshold = threshold

    def validate(self, input_text: str) -> GuardrailOutput:
        """
        Classifies text based on DuoGuard categories.

        Args:
            input_text: text that you want to validate.
        Returns:
            Whether the output is generally true (bool) and dictionary object with classifications for each category supported by
            DuoGuard.
        """
        input_tensors = self._pre_processing(input_text)
        prob_vector = self._inference(input_tensors)
        overall_label, predicted_labels = self._post_processing(prob_vector)
        return GuardrailOutput(unsafe=overall_label, explanation=predicted_labels)

    def _load_model(self) -> None:
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)  # type: ignore[no-untyped-call]
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
        self.model = model
        self.tokenizer = tokenizer

    def _pre_processing(self, input_text: str) -> Any:
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
        )
        return inputs

    def _inference(self, inputs: torch.Tensor) -> List[float]:
        """
        Processes the input text to obtain probabilities for each of the DuoGuard categories. It does this by looking at the
        logits for each category name, and then converting the logits to probabilities. These probabilities will then be used
        to determine whether, given a threshold, the input text violates any of the safety categories.

        Args:
            input_text: text that you want to validate.
        Returns:
            A list of probabilities for each category.
        """
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.sigmoid(logits)
        prob_vector = probabilities[0].tolist()
        return prob_vector

    def _post_processing(self, prob_vector: List[float]) -> Tuple[bool, Dict[str, bool]]:
        """
        Performs the decision function, as described in the DuoGuard paper (see the huggingface documentation). Can be
        overridden to define a new decision function using the probability vector.

        Args:
            prob_vector: a list of each probability that a category has been violated
        Returns:
            overall_label: True if one of the safety categories is violated, False otherwise
            predicted_labels: A mapping between the categories and which are violated. Follows the same schema as overall_label.
        """
        categories = DUOGUARD_CATEGORIES
        predicted_labels = {}
        for cat_name, prob in zip(categories, prob_vector):
            label = prob > self.threshold
            predicted_labels[cat_name] = label
        max_prob = max(prob_vector)
        overall_label = max_prob > self.threshold
        return overall_label, predicted_labels
