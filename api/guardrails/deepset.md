# Deepset

Wrapper for prompt injection detection model from Deepset.

For more information, please see the model card:

- [Deepset](https://huggingface.co/deepset/deberta-v3-base-injection).

## Supported Models

- `deepset/deberta-v3-base-injection`

## Constructor

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `model_id` | `str | None` | No | `None` |
| `provider` | `Optional[Provider[dict[str, Any], dict[str, Any]]]` | No | `None` |

Initialize the Deepset guardrail.

## validate

Default validation pipeline: preprocess -> inference -> postprocess.

**Parameters**

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `input_text` | `str` | Yes | — |

**Returns:** `GuardrailOutput`
