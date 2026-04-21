# Sentinel

Prompt injection detection encoder based model.

For more information, please see the model card:

- [Sentinel](https://huggingface.co/qualifire/prompt-injection-sentinel).

## Supported Models

- `qualifire/prompt-injection-sentinel`

## Constructor

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `model_id` | `str | None` | No | `None` |
| `provider` | `Optional[Provider[dict[str, Any], dict[str, Any]]]` | No | `None` |

Initialize the Sentinel guardrail.

## validate

Default validation pipeline: preprocess -> inference -> postprocess.

**Parameters**

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `input_text` | `str` | Yes | — |

**Returns:** `GuardrailOutput`
