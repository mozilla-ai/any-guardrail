# InjecGuard

Prompt injection detection encoder based model.

For more information, please see the model cards:

- [PIGuard](https://huggingface.co/leolee99/PIGuard) (default) — the renamed,
  maintained successor to InjecGuard; adds the "Mitigating Over-defense for
  Free" training strategy (ACL 2025). Same DeBERTa-v3 architecture and
  ``"injection"`` label, so it is a drop-in upgrade.
- [InjecGuard](https://huggingface.co/leolee99/InjecGuard) — original repo,
  kept for backward compatibility.

## Supported Models

- `leolee99/PIGuard`
- `leolee99/InjecGuard`

## Constructor

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `model_id` | `str | None` | No | `None` |
| `provider` | `Provider[dict[str, Any], dict[str, Any]] | None` | No | `None` |

Initialize the InjecGuard guardrail.

## validate

Default validation pipeline: preprocess -> inference -> postprocess.

**Parameters**

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `input_text` | `str | list[str]` | Yes | — |

**Returns:** `GuardrailOutput | list[GuardrailOutput]`
