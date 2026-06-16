"""Type definitions for the any-guardrail library.

This module provides Pydantic wrappers for preprocessing, inference, and guardrail outputs,
enabling runtime validation across all guardrail implementations.
"""

from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

# Type variables for generic stages
PreprocessT = TypeVar("PreprocessT")
"""Type variable for preprocessing output data."""

InferenceT = TypeVar("InferenceT")
"""Type variable for inference output data."""


class CategoryResult(BaseModel):
    """Result for one taxonomy category evaluated by a guardrail.

    Categories give multi-label and taxonomy guardrails (DuoGuard, Llama Guard
    S-codes, Azure severities, binary classifiers' label distributions) a
    lossless, uniform home instead of overloading ``explanation``.

    Example:
        >>> CategoryResult(name="S1", description="Violent Crimes", triggered=True)

    """

    name: str
    """Stable identifier for the category (e.g. ``"S1"``, ``"hate"``, ``"Jailbreak prompts"``)."""

    description: str | None = None
    """Human-readable elaboration (e.g. ``"Violent Crimes"`` for ``"S1"``)."""

    triggered: bool | None = None
    """Whether the guardrail flagged this category.

    Set to a concrete bool whenever the guardrail has a per-category verdict (a threshold or an
    argmax decision), so ``if c.triggered`` / ``is False`` behave uniformly. Left None only when the
    backend genuinely reports no verdict for the category (e.g. a raw score with no decision).
    """

    score: float | None = None
    """Probability-like score in ~[0, 1] for this category (for risk categories, higher = more likely violating)."""

    severity: int | None = None
    """Backend-native integer severity (e.g. Azure Content Safety 0-7). None when not reported."""


class SpanResult(BaseModel):
    """A character span flagged by a guardrail.

    Reserved for span-producing guardrails (PII/NER detection, span-level
    toxicity, token classification). Offsets are zero-based character indices
    into the validated text.
    """

    start: int
    """Zero-based start offset of the span in the validated text."""

    end: int
    """Zero-based end offset (exclusive) of the span in the validated text."""

    text: str | None = None
    """The flagged text, when the guardrail surfaces it."""

    label: str | None = None
    """Label attached to the span (e.g. an entity or category name)."""

    score: float | None = None
    """Probability-like risk score in ~[0, 1] for this span. None when not reported."""


class GuardrailUsage(BaseModel):
    """Provenance and cost envelope for one ``validate()`` call."""

    model_id: str | None = None
    """Identifier of the model or service that produced the result."""

    latency_ms: float | None = None
    """Wall-clock duration of the validation call in milliseconds."""

    prompt_tokens: int | None = None
    """Prompt token count, when the backend surfaces it."""

    completion_tokens: int | None = None
    """Completion token count, when the backend surfaces it."""


class GuardrailOutput(BaseModel):
    """Represents the output of a guardrail evaluation with runtime validation.

    This is the single concrete result shape shared by every guardrail
    implementation, so application code can swap guardrails without changes.

    Example:
        >>> output = GuardrailOutput(valid=True, explanation="Content is safe", score=0.05)
        >>> output.valid
        True

    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    valid: bool
    """Verdict: True when the content passed the guardrail. Every guardrail commits to a verdict."""

    explanation: str | None = None
    """Human-readable rationale only (judge reasoning, raw generation). Structured data lives in categories/extra."""

    score: float | None = None
    """Canonical risk score: ~[0, 1], higher = more likely violating. None when no meaningful risk score exists."""

    categories: list[CategoryResult] = Field(default_factory=list)
    """Per-category results. Empty when the guardrail has no category taxonomy."""

    spans: list[SpanResult] | None = None
    """Character spans flagged by the guardrail. None when the guardrail does not produce spans."""

    modified_text: str | None = None
    """Sanitized/masked text when the guardrail rewrites input. None means no modification."""

    action: str | None = None
    """Provider-recommended enforcement action (e.g. ``"block"``), when the backend returns one.

    Free-form and provider-native (any-guardrail does not impose an action vocabulary). Advisory
    only: enforcement is the caller's responsibility. None when the backend recommends no action.
    """

    usage: GuardrailUsage | None = None
    """Provenance and cost information for this validation call."""

    extra: dict[str, Any] | None = None
    """Structured, guardrail-specific extras (e.g. rubric score, highlights, blocklist matches).

    Must be JSON-serializable so the output round-trips through ``model_dump_json()`` and matches
    the published schema. Push non-serializable backend/SDK objects to ``raw`` instead.
    """

    raw: Any | None = None
    """Unmodified backend payload (API response JSON, model tensors) for escape-hatch access."""


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
AnyDict = dict[str, Any]
"""Type alias for generic dictionary with string keys and any values."""

TokenizerDict = AnyDict
"""Type alias for tokenizer output dictionaries (alias for AnyDict)."""

ChatMessage = dict[str, str]
"""Type alias for a single chat message with role and content."""

ChatMessages = list[ChatMessage]
"""Type alias for a list of chat messages."""

# Standard type aliases for common guardrail patterns
StandardPreprocessOutput = GuardrailPreprocessOutput[AnyDict]
"""Type alias for standard preprocessing output with AnyDict data."""

StandardInferenceOutput = GuardrailInferenceOutput[AnyDict]
"""Type alias for standard inference output with AnyDict data."""
