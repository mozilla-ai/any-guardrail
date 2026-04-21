# Pangolin

Prompt injection detection encoder based models.

For more information, please see the model card:

- [Pangolin Base](https://huggingface.co/dcarpintero/pangolin-guard-base)

## Supported Models

- `dcarpintero/pangolin-guard-base`

## Constructor

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `model_id` | `str | None` | No | `None` |
| `provider` | `Optional[Provider[dict[str, Any], dict[str, Any]]]` | No | `None` |

Initialize the Pangolin guardrail.

## validate

Default validation pipeline: preprocess -> inference -> postprocess.

**Parameters**

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `input_text` | `str` | Yes | — |

**Returns:** `GuardrailOutput`
