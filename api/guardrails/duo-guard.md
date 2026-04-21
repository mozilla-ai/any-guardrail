# DuoGuard

Guardrail that classifies text based on the categories in DUOGUARD_CATEGORIES.

For more information, please see the model card:

- [DuoGuard](https://huggingface.co/collections/DuoGuard/duoguard-models-67a29ad8bd579a404e504d21).

## Supported Models

- `DuoGuard/DuoGuard-0.5B`
- `DuoGuard/DuoGuard-1B-Llama-3.2-transfer`
- `DuoGuard/DuoGuard-1.5B-transfer`

## Constructor

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `model_id` | `str | None` | No | `None` |
| `threshold` | `float` | No | `0.5` |
| `provider` | `Optional[Provider[dict[str, Any], dict[str, Any]]]` | No | `None` |

Initialize the DuoGuard model.

## validate

Default validation pipeline: preprocess -> inference -> postprocess.

**Parameters**

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `input_text` | `str` | Yes | — |

**Returns:** `GuardrailOutput`
