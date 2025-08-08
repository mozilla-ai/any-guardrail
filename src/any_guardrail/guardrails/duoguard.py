from any_guardrail.guardrails.guardrail import Guardrail
from any_guardrail.utils.custom_types import GuardrailOutput, GuardrailModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, PreTrainedModel
import torch
from typing import Tuple, Dict, List

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

    def __init__(self, model_id: str, threshold: float = DUOGUARD_DEFAULT_THRESHOLD) -> None:
        super().__init__(model_id)
        if self.model_id in [
            "DuoGuard/DuoGuard-0.5B",
            "DuoGuard/DuoGuard-1B-Llama-3.2-transfer",
            "DuoGuard/DuoGuard-1.5B-transfer",
        ]:
            self.guardrail = self._model_instantiation()
        else:
            raise ValueError(
                "Must instantiate model using one of the following paths: "
                "\n\n DuoGuard/DuoGuard-0.5B \n DuoGuard/DuoGuard-1B-Llama-3.2-transfer \n DuoGuard/DuoGuard-1.5B-transfer"
            )
        self.threshold = threshold

    def validate(self, input_text: str) -> GuardrailOutput:
        """
        Classifies text based on DuoGuard categories.

        Args:
            input_text: text that you want to safety_review.
        Returns:
            Whether the output is generally true (bool) and dictionary object with classifications for each category supported by
            DuoGuard.
        """
        prob_vector = self._get_probabilities(input_text)
        overall_label, predicted_labels = self._classification_decision(prob_vector)
        return GuardrailOutput(unsafe=overall_label, explanation=predicted_labels)

    def _model_instantiation(self) -> GuardrailModel:
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)  # type: ignore[no-untyped-call]
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
        return GuardrailModel(model=model, tokenizer=tokenizer)

    def _get_probabilities(self, input_text: str) -> List[float]:
        """
        Processes the input text to obtain probabilities for each of the DuoGuard categories. It does this by looking at the
        logits for each category name, and then converting the logits to probabilities. These probabilities will then be used
        to determine whether, given a threshold, the input text violates any of the safety categories.

        Args:
            input_text: text that you want to safety_review.
        Returns:
            A list of probabilities for each category.
        """
        if self.guardrail.tokenizer:
            inputs = self.guardrail.tokenizer(
                input_text,
                return_tensors="pt",
            )
            if isinstance(self.guardrail.model, PreTrainedModel):
                with torch.no_grad():
                    outputs = self.guardrail.model(**inputs)
                    logits = outputs.logits
                    probabilities = torch.sigmoid(logits)
                prob_vector = probabilities[0].tolist()
            else:
                raise TypeError("Using the incorrect model type for DuoGuard")
            return prob_vector
        else:
            raise TypeError("Did not instantiate tokenizer.")

    def _classification_decision(self, prob_vector: List[float]) -> Tuple[bool, Dict[str, bool]]:
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
