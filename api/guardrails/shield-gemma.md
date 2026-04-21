# ShieldGemma

Wrapper class for Google ShieldGemma models.

For more information, please visit the model cards: [Shield Gemma](https://huggingface.co/collections/google/shieldgemma-67d130ef8da6af884072a789).

Note we do not support the image classifier.

## Supported Models

- `google/shieldgemma-2b`
- `google/shieldgemma-9b`
- `google/shieldgemma-27b`

## Constructor

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `policy` | `str` | Yes | — |
| `threshold` | `float` | No | `0.5` |
| `model_id` | `str | None` | No | `None` |
| `provider` | `Optional[Provider[dict[str, Any], dict[str, Any]]]` | No | `None` |

Initialize the ShieldGemma guardrail.

## validate

Default validation pipeline: preprocess -> inference -> postprocess.

**Parameters**

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `input_text` | `str` | Yes | — |

**Returns:** `GuardrailOutput`
