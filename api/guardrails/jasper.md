# Jasper

Prompt injection detection encoder based models.

For more information, please see the model card:

- [Jasper Deberta](https://huggingface.co/JasperLS/deberta-v3-base-injection)
- [Jasper Gelectra](https://huggingface.co/JasperLS/gelectra-base-injection).

Args:
    model_id: HuggingFace path to model.

Raises:
    ValueError: Can only use model paths for Jasper models from HuggingFace.

## Supported Models

- `JasperLS/gelectra-base-injection`
- `JasperLS/deberta-v3-base-injection`

## Constructor

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `model_id` | `str | None` | No | `None` |
| `provider` | `Optional[Provider[dict[str, Any], dict[str, Any]]]` | No | `None` |

Initialize the Jasper guardrail.

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
