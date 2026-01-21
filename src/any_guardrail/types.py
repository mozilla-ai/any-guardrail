"""Type definitions for the any-guardrail library.

This module provides Pydantic wrappers for preprocessing, inference, and guardrail outputs,
enabling runtime validation across all guardrail implementations.
"""

from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict

# Type variables for generic stages
PreprocessT = TypeVar("PreprocessT")
"""Type variable for preprocessing output data."""

InferenceT = TypeVar("InferenceT")
"""Type variable for inference output data."""

ValidT = TypeVar("ValidT", default=Any)
"""Type variable for guardrail output valid field."""

ExplanationT = TypeVar("ExplanationT", default=Any)
"""Type variable for guardrail output explanation field."""

ScoreT = TypeVar("ScoreT", default=Any)
"""Type variable for guardrail output score field."""


class GuardrailOutput(BaseModel, Generic[ValidT, ExplanationT, ScoreT]):
    """Represents the output of a guardrail evaluation with runtime validation.

    This class wraps the final output of the guardrail evaluation, providing
    a consistent interface and runtime validation across all guardrail
    implementations.

    Type Parameters:
        ValidT: The type of the valid field (e.g., bool, str, custom enum).
        ExplanationT: The type of the explanation field (e.g., str, dict, list).
        ScoreT: The type of the score field (e.g., float, int, dict).

    Example:
        >>> output = GuardrailOutput(valid=True, explanation="Content is safe", score=0.95)
        >>> output.valid
        True

    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    valid: ValidT | None = None
    """Indicates if the output should be considered valid (safe/acceptable)."""

    explanation: ExplanationT | None = None
    """Provides an explanation for the guardrail evaluation result."""

    score: ScoreT | None = None
    """Represents the score assigned to the output by the guardrail."""


class GuardrailPreprocessOutput(BaseModel, Generic[PreprocessT]):
    """Wrapper for preprocessing outputs with runtime validation.

    This class wraps the output of the preprocessing stage, providing
    runtime validation and a consistent interface across all guardrail
    implementations.

    Type Parameters:
        PreprocessT: The type of the preprocessing result (e.g., tokenized input,
            API options, chat messages).

    Example:
        >>> output = GuardrailPreprocessOutput(data={"input_ids": tensor, "attention_mask": tensor})
        >>> output.data["input_ids"]

    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: PreprocessT
    """The preprocessing result (tokenized input, API options, etc.)."""


class GuardrailInferenceOutput(BaseModel, Generic[InferenceT]):
    """Wrapper for inference outputs with runtime validation.

    This class wraps the output of the inference stage, providing
    runtime validation and a consistent interface across all guardrail
    implementations.

    Type Parameters:
        InferenceT: The type of the inference result (e.g., model logits,
            API response, generated tokens).

    Example:
        >>> output = GuardrailInferenceOutput(data=model_output)
        >>> logits = output.data["logits"]

    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: InferenceT
    """The inference result (model logits, API response, etc.)."""


# Type aliases for common patterns
TokenizerDict = dict[str, Any]
"""Type alias for tokenizer output dictionaries."""

ChatMessage = dict[str, str]
"""Type alias for a single chat message with role and content."""

ChatMessages = list[ChatMessage]
"""Type alias for a list of chat messages."""
