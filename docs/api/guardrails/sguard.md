# Sguard

Samsung SDS SGuard-v1 — multilingual content-safety and jailbreak filters.

Two 2B Granite-based guards selectable by ``model_id``:

- **ContentFilter** classifies prompt/response against five categories
  (Crime, Manipulation, Privacy, Sexual, Violence).
- **JailbreakFilter** flags jailbreak-framed inputs (binary safe/unsafe).

``valid`` is ``True`` when nothing is flagged. Fails closed (``valid=False`` with
``extra={"parse_failure": True}``) when the output cannot be read.

Note: SGuard natively emits per-category safe/unsafe *tokens* read from logits; this
integration parses the decoded text and is best-effort — validate against the model
on real weights before production use.

For more information, see the model cards:

- [SGuard-ContentFilter-2B-v1](https://huggingface.co/SamsungSDS-Research/SGuard-ContentFilter-2B-v1) (default).
- [SGuard-JailbreakFilter-2B-v1](https://huggingface.co/SamsungSDS-Research/SGuard-JailbreakFilter-2B-v1).

Args:
    model_id: Optional HuggingFace model ID. Defaults to the ContentFilter model.
    provider: Optional pre-configured provider. Defaults to a ``HuggingFaceProvider``
        loading a causal LM.

## Supported Models

- `SamsungSDS-Research/SGuard-ContentFilter-2B-v1`
- `SamsungSDS-Research/SGuard-JailbreakFilter-2B-v1`

## Constructor

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `model_id` | `str | None` | No | `None` |
| `provider` | `Provider[dict[str, Any], dict[str, Any]] | None` | No | `None` |

Initialize the SGuard guardrail.

## validate

Classify ``input_text`` (and optionally an assistant ``output_text``).

**Parameters**

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `input_text` | `str` | Yes | — |
| `output_text` | `str | None` | No | `None` |

**Returns:** `GuardrailOutput`
