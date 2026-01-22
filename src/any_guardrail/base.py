from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, ClassVar, Generic

from any_guardrail.types import (
    ExplanationT,
    GuardrailInferenceOutput,
    GuardrailOutput,
    GuardrailPreprocessOutput,
    InferenceT,
    PreprocessT,
    ScoreT,
    ValidT,
)

__all__ = [
    "Guardrail",
    "GuardrailName",
    "GuardrailOutput",
    "ThreeStageGuardrail",
]


class GuardrailName(str, Enum):
    """String enum for supported guardrails."""

    ANYLLM = "any_llm"
    DEEPSET = "deepset"
    DUOGUARD = "duo_guard"
    FLOWJUDGE = "flowjudge"
    GLIDER = "glider"
    HARMGUARD = "harm_guard"
    INJECGUARD = "injec_guard"
    JASPER = "jasper"
    OFFTOPIC = "off_topic"
    PANGOLIN = "pangolin"
    PROTECTAI = "protectai"
    SENTINEL = "sentinel"
    SHIELD_GEMMA = "shield_gemma"
    LLAMA_GUARD = "llama_guard"
    AZURE_CONTENT_SAFETY = "azure_content_safety"


class Guardrail(ABC, Generic[ValidT, ExplanationT, ScoreT]):
    """Base class for all guardrails."""

    SUPPORTED_MODELS: ClassVar[list[str]] = []

    @abstractmethod
    def validate(self, *args: Any, **kwargs: Any) -> GuardrailOutput[ValidT, ExplanationT, ScoreT]:
        """Abstract method for validating some input. Each subclass implements its own signature."""
        msg = "Each subclass will create their own method."
        raise NotImplementedError(msg)


class ThreeStageGuardrail(
    Guardrail[ValidT, ExplanationT, ScoreT], ABC, Generic[PreprocessT, InferenceT, ValidT, ExplanationT, ScoreT]
):
    """Base class for guardrails using preprocess -> inference -> postprocess with runtime validation.

    This abstract class provides a structured pipeline for guardrail implementations
    that follow the three-stage pattern. Each stage uses Pydantic wrappers for
    runtime type validation.

    Type Parameters:
        PreprocessT: The type of data produced by preprocessing (e.g., tokenized input,
            API options).
        InferenceT: The type of data produced by inference (e.g., model logits,
            API response).

    Example:
        >>> class MyGuardrail(ThreeStageGuardrail[dict[str, Any], dict[str, Any]]):
        ...     def _pre_processing(self, text: str) -> GuardrailPreprocessOutput[dict[str, Any]]:
        ...         return GuardrailPreprocessOutput(data={"text": text})
        ...
        ...     def _inference(self, inputs: GuardrailPreprocessOutput[dict[str, Any]]) -> GuardrailInferenceOutput[dict[str, Any]]:
        ...         result = self.model(inputs.data)
        ...         return GuardrailInferenceOutput(data=result)
        ...
        ...     def _post_processing(self, outputs: GuardrailInferenceOutput[dict[str, Any]]) -> GuardrailOutput:
        ...         return GuardrailOutput(valid=outputs.data["score"] > 0.5)

    """

    @abstractmethod
    def _pre_processing(self, *args: Any, **kwargs: Any) -> GuardrailPreprocessOutput[PreprocessT]:
        """Transform input into format for inference.

        Args:
            *args: Input arguments (implementation-specific).
            **kwargs: Additional keyword arguments.

        Returns:
            GuardrailPreprocessOutput wrapping the preprocessing result.

        """
        ...

    @abstractmethod
    def _inference(self, model_inputs: GuardrailPreprocessOutput[PreprocessT]) -> GuardrailInferenceOutput[InferenceT]:
        """Run the core inference/API call.

        Args:
            model_inputs: The wrapped preprocessing output.

        Returns:
            GuardrailInferenceOutput wrapping the inference result.

        """
        ...

    @abstractmethod
    def _post_processing(
        self, model_outputs: GuardrailInferenceOutput[InferenceT]
    ) -> GuardrailOutput[ValidT, ExplanationT, ScoreT]:
        """Transform inference output to GuardrailOutput.

        Args:
            model_outputs: The wrapped inference output.

        Returns:
            GuardrailOutput with valid, explanation, and/or score fields.

        """
        ...
