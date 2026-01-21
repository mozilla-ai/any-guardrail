from abc import ABC, abstractmethod
from typing import Any, Generic

try:
    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    MISSING_PACKAGES_ERROR = None

except ImportError as e:
    MISSING_PACKAGES_ERROR = e

from any_guardrail.base import Guardrail, GuardrailOutput
from any_guardrail.types import (
    GuardrailInferenceOutput,
    GuardrailPreprocessOutput,
    InferenceT,
    PreprocessT,
)


def _softmax(_outputs):  # type: ignore[no-untyped-def]
    maxes = np.max(_outputs, axis=-1, keepdims=True)
    shifted_exp = np.exp(_outputs - maxes)
    return shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)


def _match_injection_label(
    model_outputs: GuardrailInferenceOutput[dict[str, Any]], injection_label: str, id2label: dict[int, str]
) -> GuardrailOutput[bool, None, float]:
    """Match injection label from model outputs.

    Args:
        model_outputs: The wrapped inference output containing logits.
        injection_label: The label indicating injection/unsafe content.
        id2label: Mapping from label IDs to label names.

    Returns:
        GuardrailOutput with valid=True if content is safe, valid=False if injection detected.

    """
    logits = model_outputs.data["logits"][0].numpy()
    scores = _softmax(logits)  # type: ignore[no-untyped-call]
    label = id2label[scores.argmax().item()]
    return GuardrailOutput(valid=label != injection_label, score=scores.max().item())


class HuggingFace(Guardrail, ABC, Generic[PreprocessT, InferenceT]):
    """Wrapper for models from Hugging Face with typed preprocessing and inference stages.

    This base class provides a three-stage pipeline (preprocess -> inference -> postprocess)
    with runtime type validation via Pydantic wrappers.

    Type Parameters:
        PreprocessT: The type of data produced by preprocessing. Defaults to dict[str, Any]
            for standard tokenizer output.
        InferenceT: The type of data produced by inference. Defaults to dict[str, Any]
            for standard model output with logits.

    Example:
        >>> class MyGuardrail(HuggingFace[dict[str, Any], dict[str, Any]]):
        ...     SUPPORTED_MODELS = ["my-model"]
        ...
        ...     def _post_processing(self, model_outputs: GuardrailInferenceOutput[dict[str, Any]]) -> GuardrailOutput:
        ...         logits = model_outputs.data["logits"]
        ...         return GuardrailOutput(valid=logits[0] > 0)

    """

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

    def validate(self, input_text: str) -> GuardrailOutput[Any, Any, Any]:
        """Validate whether the input text is safe or not."""
        model_inputs = self._pre_processing(input_text)
        model_outputs = self._inference(model_inputs)
        return self._post_processing(model_outputs)

    def _load_model(self) -> None:
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)  # type: ignore[no-untyped-call]

    def _pre_processing(self, input_text: str) -> GuardrailPreprocessOutput[PreprocessT]:
        """Preprocess input text into model inputs.

        Args:
            input_text: The text to preprocess.

        Returns:
            GuardrailPreprocessOutput wrapping the tokenized input.

        """
        tokenized = self.tokenizer(input_text, return_tensors="pt")
        return GuardrailPreprocessOutput(data=tokenized)

    def _inference(self, model_inputs: GuardrailPreprocessOutput[PreprocessT]) -> GuardrailInferenceOutput[InferenceT]:
        """Run model inference on preprocessed inputs.

        Args:
            model_inputs: The wrapped preprocessing output.

        Returns:
            GuardrailInferenceOutput wrapping the model output.

        """
        with torch.no_grad():
            output = self.model(**model_inputs.data)
        return GuardrailInferenceOutput(data=output)

    @abstractmethod
    def _post_processing(self, model_outputs: GuardrailInferenceOutput[InferenceT]) -> GuardrailOutput[Any, Any, Any]:
        """Process the model outputs to return a GuardrailOutput.

        Args:
            model_outputs: The wrapped inference output.

        Returns:
            GuardrailOutput: The processed output indicating safety or other metrics.

        """
        msg = "Each subclass will create their own method."
        raise NotImplementedError(msg)
