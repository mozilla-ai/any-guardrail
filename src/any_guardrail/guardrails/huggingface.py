from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from any_guardrail.guardrail import Guardrail
from any_guardrail.types import GuardrailOutput


def _softmax(_outputs):
    maxes = np.max(_outputs, axis=-1, keepdims=True)
    shifted_exp = np.exp(_outputs - maxes)
    return shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)


def _match_injection_label(
    model_outputs: list[dict[str, str | float]], injection_label: str, id2label: dict[int, str]
) -> GuardrailOutput:
    logits = model_outputs["logits"][0].numpy()
    scores = _softmax(logits)
    label = id2label[scores.argmax().item()]
    return GuardrailOutput(unsafe=label == injection_label, score=scores.max().item())


class HuggingFace(Guardrail, ABC):
    """Wrapper for models from Hugging Face."""

    SUPPORTED_MODELS: ClassVar[list[str]] = []

    def __init__(self, model_id: str | None = None) -> None:
        """Initialize the guardrail with a model ID."""
        if model_id is None:
            model_id = self.SUPPORTED_MODELS[0] if self.SUPPORTED_MODELS else None
        self.model_id = model_id
        self._validate_model_id(model_id)
        self._load_model()

    def _validate_model_id(self, model_id: str) -> None:
        if model_id not in self.SUPPORTED_MODELS:
            msg = f"Only supports {self.SUPPORTED_MODELS}. Please use this path to instantiate model."
            raise ValueError(msg)

    def validate(self, input_text: str) -> GuardrailOutput:
        """Validate whether the input text is safe or not."""
        model_inputs = self._pre_processing(input_text)
        model_outputs = self._inference(model_inputs)
        return self._post_processing(model_outputs)

    def _load_model(self) -> None:
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

    def _pre_processing(self, input_text: str) -> torch.Tensor:
        return self.tokenizer(input_text, return_tensors="pt")

    def _inference(self, model_inputs: torch.Tensor) -> list[dict[str, str | float]]:
        with torch.no_grad():
            return self.model(**model_inputs)

    @abstractmethod
    def _post_processing(self, model_outputs: list[dict[str, str | float]]) -> GuardrailOutput:
        """Process the model outputs to return a GuardrailOutput.

        Args:
            model_outputs: The outputs from the model inference.

        Returns:
            GuardrailOutput: The processed output indicating safety or other metrics.

        """
        msg = "Each subclass will create their own method."
        raise NotImplementedError(msg)
