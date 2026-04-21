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

**Parameters**

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `input_text` | `str` | Yes | — |

**Returns:** `GuardrailOutput`
