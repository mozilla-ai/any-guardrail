# Types

Runtime-validated wrappers used throughout the pipeline and the output type returned by every guardrail.

## GuardrailOutput

Represents the output of a guardrail evaluation with runtime validation.

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

| Field | Type | Description |
|-------|------|-------------|
| `valid` | `Optional[~ValidT]` |  |
| `explanation` | `Optional[~ExplanationT]` |  |
| `score` | `Optional[~ScoreT]` |  |

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
