# Pangolin

Prompt injection detection encoder based models.

For more information, please see the model cards:

- [Pangolin Base](https://huggingface.co/dcarpintero/pangolin-guard-base) (default) — ModernBERT-base.
- [Pangolin Large](https://huggingface.co/dcarpintero/pangolin-guard-large) — ModernBERT-large;
  higher accuracy, 8192-token context. Same ``"unsafe"`` label, so it is a drop-in alternative.

## Supported Models

- `dcarpintero/pangolin-guard-base`
- `dcarpintero/pangolin-guard-large`

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
| `input_text` | `str | list[str]` | Yes | — |

**Returns:** `GuardrailOutput | list[GuardrailOutput]`
