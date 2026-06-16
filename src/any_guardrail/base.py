import time
from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any, ClassVar, Generic, overload

from any_guardrail.types import (
    AnyDict,
    GuardrailInferenceOutput,
    GuardrailOutput,
    GuardrailPreprocessOutput,
    GuardrailUsage,
    InferenceT,
    PreprocessT,
)

__all__ = [
    "Guardrail",
    "GuardrailName",
    "GuardrailOutput",
    "ThreeStageGuardrail",
]


class GuardrailName(StrEnum):
    """String enum for supported guardrails."""

    ANYLLM = "any_llm"
    BEDROCK_GUARDRAILS = "bedrock_guardrails"
    DEEPSET = "deepset"
    DUOGUARD = "duo_guard"
    FLOWJUDGE = "flowjudge"
    GLIDER = "glider"
    GRANITE_GUARDIAN = "granite_guardian"
    HARMGUARD = "harm_guard"
    INJECGUARD = "injec_guard"
    JASPER = "jasper"
    OFFTOPIC = "off_topic"
    OPENAI_MODERATION = "openai_moderation"
    PANGOLIN = "pangolin"
    PROTECTAI = "protectai"
    SENTINEL = "sentinel"
    SHIELD_GEMMA = "shield_gemma"
    LLAMA_GUARD = "llama_guard"
    AZURE_CONTENT_SAFETY = "azure_content_safety"
    AZURE_PROMPT_SHIELDS = "azure_prompt_shields"
    ALINIA = "alinia"
    LAKERA_GUARD = "lakera_guard"
    PROMPT_GUARD = "prompt_guard"
    BIELIK_GUARD = "bielik_guard"
    WILD_GUARD = "wild_guard"
    DYNA_GUARD = "dyna_guard"
    SGUARD = "sguard"
    NEMOTRON_CONTENT_SAFETY = "nemotron_content_safety"
    POLY_GUARD = "poly_guard"
    KANANA_SAFEGUARD = "kanana_safeguard"
    GPT_OSS_SAFEGUARD = "gpt_oss_safeguard"
    PROMETHEUS = "prometheus"
    COMPASS_JUDGER = "compass_judger"
    SELENE = "selene"
    PRIVACY_FILTER = "privacy_filter"
    LETTUCE_DETECT = "lettuce_detect"
    HHEM = "hhem"
    GLI_GUARD = "gli_guard"


class Guardrail(ABC):
    """Base class for all guardrails."""

    SUPPORTED_MODELS: ClassVar[list[str]] = []

    @abstractmethod
    def validate(self, *args: Any, **kwargs: Any) -> GuardrailOutput:
        """Abstract method for validating some input. Each subclass implements its own signature."""
        msg = "Each subclass will create their own method."
        raise NotImplementedError(msg)

    def _stamp_usage(self, result: GuardrailOutput, latency_ms: float) -> None:
        """Fill provenance fields on ``result.usage`` that the guardrail left unset.

        Merge semantics: only ``None`` fields are filled, so guardrails that
        already record token counts or a custom model id keep their values.
        """
        usage = result.usage if result.usage is not None else GuardrailUsage()
        if usage.model_id is None:
            model_id = getattr(self, "model_id", None)
            usage.model_id = model_id if isinstance(model_id, str) else None
        if usage.latency_ms is None:
            usage.latency_ms = latency_ms
        result.usage = usage


class ThreeStageGuardrail(Guardrail, ABC, Generic[PreprocessT, InferenceT]):
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
    def _post_processing(self, model_outputs: GuardrailInferenceOutput[InferenceT]) -> GuardrailOutput:
        """Transform inference output to GuardrailOutput.

        Args:
            model_outputs: The wrapped inference output.

        Returns:
            GuardrailOutput with valid, explanation, and/or score fields.

        """
        ...

    @overload
    def validate(self, input_text: str, **kwargs: Any) -> GuardrailOutput: ...

    @overload
    def validate(self, input_text: list[str], **kwargs: Any) -> list[GuardrailOutput]: ...

    def validate(self, input_text: str | list[str], **kwargs: Any) -> GuardrailOutput | list[GuardrailOutput]:
        """Default validation pipeline: preprocess -> inference -> postprocess.

        Args:
            input_text: The text to validate. If a list is supplied, each item is
                validated and a list of GuardrailOutputs is returned in the same
                order. Subclasses can override ``_validate_batch`` to enable true
                batched inference; the default iterates over inputs.
            **kwargs: Additional arguments passed to preprocessing (e.g., output_text, comparison_text).

        Returns:
            GuardrailOutput when ``input_text`` is a string, or a list of
            GuardrailOutputs when ``input_text`` is a list.

        Note:
            Subclasses can override this method to customize the signature or add validation logic.

        """
        if isinstance(input_text, list):
            start = time.perf_counter()
            results = self._validate_batch(input_text, **kwargs)
            # Whole-batch wall-clock: a true-batch _validate_batch runs one shared
            # inference call, so per-item latency isn't measurable. Every item is
            # stamped with the batch duration (documented on GuardrailUsage.latency_ms).
            latency_ms = (time.perf_counter() - start) * 1000.0
            for result in results:
                self._stamp_usage(result, latency_ms)
            return results
        return self._execute(input_text, **kwargs)

    def _execute(self, *args: Any, **kwargs: Any) -> GuardrailOutput:
        """Run the three-stage pipeline and stamp provenance on the result.

        Subclasses that override ``validate`` with a custom signature should
        route their pipeline through this method so ``usage`` (model_id,
        latency_ms) is populated uniformly.
        """
        start = time.perf_counter()
        model_inputs = self._pre_processing(*args, **kwargs)
        model_outputs = self._inference(model_inputs)
        result = self._post_processing(model_outputs)
        self._stamp_usage(result, (time.perf_counter() - start) * 1000.0)
        return result

    def _validate_batch(self, input_texts: list[str], **kwargs: Any) -> list[GuardrailOutput]:
        """Validate a batch of inputs.

        Default implementation iterates over the list and validates each item
        independently. Subclasses that can perform true batched inference (e.g.
        encoder classifiers via padded tokenization) should override this method
        to take advantage of vectorized inference.

        Args:
            input_texts: The texts to validate.
            **kwargs: Additional keyword arguments passed to each ``validate`` call.

        Returns:
            A list of GuardrailOutputs, one per input, in the same order.

        """
        results: list[GuardrailOutput] = []
        for text in input_texts:
            result = self.validate(text, **kwargs)
            if isinstance(result, list):
                msg = "validate() returned a list for a single string input; check subclass overrides."
                raise TypeError(msg)
            results.append(result)
        return results


# Standard guardrail type alias
StandardGuardrail = ThreeStageGuardrail[AnyDict, AnyDict]
"""Type alias for guardrails whose preprocessing and inference stages exchange AnyDict payloads."""
