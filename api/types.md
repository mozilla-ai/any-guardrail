# Types

Runtime-validated wrappers used throughout the pipeline and the output type returned by every guardrail.

A machine-readable JSON Schema for `GuardrailOutput` (generated from these models) is published at <https://raw.githubusercontent.com/mozilla-ai/any-guardrail/main/schemas/guardrail_output.schema.json>. Pin a release tag in the URL for a specific version.

## GuardrailOutput

Represents the output of a guardrail evaluation with runtime validation.

This is the single concrete result shape shared by every guardrail
implementation, so application code can swap guardrails without changes.

Example:
    >>> output = GuardrailOutput(valid=True, explanation="Content is safe", score=0.05)
    >>> output.valid
    True

| Field | Type | Description |
|-------|------|-------------|
| `valid` | `bool` |  |
| `explanation` | `str | None` |  |
| `score` | `float | None` |  |
| `categories` | `list[CategoryResult]` |  |
| `spans` | `list[SpanResult] | None` |  |
| `modified_text` | `str | None` |  |
| `action` | `str | None` |  |
| `usage` | `GuardrailUsage | None` |  |
| `extra` | `dict[str, Any] | None` |  |
| `raw` | `Any | None` |  |

## CategoryResult

Result for one taxonomy category evaluated by a guardrail.

Categories give multi-label and taxonomy guardrails (DuoGuard, Llama Guard
S-codes, Azure severities, binary classifiers' label distributions) a
lossless, uniform home instead of overloading ``explanation``.

Example:
    >>> CategoryResult(name="S1", description="Violent Crimes", triggered=True)

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` |  |
| `description` | `str | None` |  |
| `triggered` | `bool | None` |  |
| `score` | `float | None` |  |
| `severity` | `int | None` |  |

## SpanResult

A character span flagged by a guardrail.

Reserved for span-producing guardrails (PII/NER detection, span-level
toxicity, token classification). Offsets are zero-based character indices
into the validated text.

| Field | Type | Description |
|-------|------|-------------|
| `start` | `int` |  |
| `end` | `int` |  |
| `text` | `str | None` |  |
| `label` | `str | None` |  |
| `score` | `float | None` |  |

## GuardrailUsage

Provenance and cost envelope for one ``validate()`` call.

| Field | Type | Description |
|-------|------|-------------|
| `model_id` | `str | None` |  |
| `latency_ms` | `float | None` |  |
| `prompt_tokens` | `int | None` |  |
| `completion_tokens` | `int | None` |  |

## GuardrailPreprocessOutput

Wrapper for preprocessing outputs with runtime validation.

This class wraps the output of the preprocessing stage, providing
runtime validation and a consistent interface across all guardrail
implementations.

Type Parameters:
    PreprocessT: The type of the preprocessing result (e.g., tokenized input,
        API options, chat messages).

Example:
    >>> output = GuardrailPreprocessOutput(data={"input_ids": tensor, "attention_mask": tensor})
    >>> output.data["input_ids"]

| Field | Type | Description |
|-------|------|-------------|
| `data` | `~PreprocessT` |  |

## GuardrailInferenceOutput

Wrapper for inference outputs with runtime validation.

This class wraps the output of the inference stage, providing
runtime validation and a consistent interface across all guardrail
implementations.

Type Parameters:
    InferenceT: The type of the inference result (e.g., model logits,
        API response, generated tokens).

Example:
    >>> output = GuardrailInferenceOutput(data=model_output)
    >>> logits = output.data["logits"]

| Field | Type | Description |
|-------|------|-------------|
| `data` | `~InferenceT` |  |
