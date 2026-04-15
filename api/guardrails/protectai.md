# Protectai

Prompt injection detection encoder based models.

For more information, please see the model card:

- [ProtectAI](https://huggingface.co/collections/protectai/llm-security-65c1f17a11c4251eeab53f40).

## Supported Models

- `ProtectAI/deberta-v3-small-prompt-injection-v2`
- `ProtectAI/distilroberta-base-rejection-v1`
- `ProtectAI/deberta-v3-base-prompt-injection`
- `ProtectAI/deberta-v3-base-prompt-injection-v2`

## Constructor

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `model_id` | `str | None` | No | `None` |
| `provider` | `Optional[Provider[dict[str, Any], dict[str, Any]]]` | No | `None` |

Initialize the Protectai guardrail.

## validate

Default validation pipeline: preprocess -> inference -> postprocess.

Args:
    input_text: The text to validate.
    **kwargs: Additional arguments passed to preprocessing (e.g., output_text, comparison_text).

Returns:
    GuardrailOutput with validation results.

Note:
    Subclasses can override this method to customize the signature or add validation logic.

**Parameters**

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `input_text` | `str` | Yes | — |

**Returns:** `GuardrailOutput`
