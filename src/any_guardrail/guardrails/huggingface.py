from abc import ABC, abstractmethod
from typing import Any

try:
    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    MISSING_PACKAGES_ERROR = None

except ImportError as e:
    MISSING_PACKAGES_ERROR = e

from any_guardrail.base import Guardrail, GuardrailOutput


def _softmax(_outputs):  # type: ignore[no-untyped-def]
    maxes = np.max(_outputs, axis=-1, keepdims=True)
    shifted_exp = np.exp(_outputs - maxes)
    return shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)


def _match_injection_label(
    model_outputs: dict[str, Any], injection_label: str, id2label: dict[int, str]
) -> GuardrailOutput:
    logits = model_outputs["logits"][0].numpy()
    scores = _softmax(logits)  # type: ignore[no-untyped-call]
    label = id2label[scores.argmax().item()]
    return GuardrailOutput(valid=label != injection_label, score=scores.max().item())


class HuggingFace(Guardrail, ABC):
    """Wrapper for models from Hugging Face."""

    def __init__(self, model_id: str | None = None) -> None:
        """Initialize the guardrail with a model ID."""
        if MISSING_PACKAGES_ERROR is not None:
            msg = "Missing packages for HuggingFace guardrail. You can try `pip install 'any-guardrail[huggingface]'`"
            raise ImportError(msg) from MISSING_PACKAGES_ERROR

        if model_id is None:
            model_id = self.SUPPORTED_MODELS[0]
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
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)  # type: ignore[no-untyped-call]

    def _pre_processing(self, input_text: str) -> Any:
        return self.tokenizer(input_text, return_tensors="pt")

    def _inference(self, model_inputs: Any) -> Any:
        with torch.no_grad():
            return self.model(**model_inputs)

    @abstractmethod
    def _post_processing(self, model_outputs: Any) -> GuardrailOutput:
        """Process the model outputs to return a GuardrailOutput.

        Args:
            model_outputs: The outputs from the model inference.

        Returns:
            GuardrailOutput: The processed output indicating safety or other metrics.

        """
        msg = "Each subclass will create their own method."
        raise NotImplementedError(msg)
