# PolyGuard

PolyGuard — multilingual safety moderation judge (17 languages).

Generative classifier reporting request harmfulness, response refusal, response
harmfulness, and the MLCommons policy categories violated. ``valid`` is ``False``
when the request or response is harmful; violated S-codes and the boolean signals
are surfaced as ``categories``. Fails closed (``valid=False`` with
``extra={"parse_failure": True}``) when no harmfulness field parses.

For more information, see the model cards:

- [PolyGuard-Ministral](https://huggingface.co/ToxicityPrompts/PolyGuard-Ministral) (default).
- [PolyGuard-Qwen](https://huggingface.co/ToxicityPrompts/PolyGuard-Qwen).
- [PolyGuard-Qwen-Smol](https://huggingface.co/ToxicityPrompts/PolyGuard-Qwen-Smol).

Args:
    model_id: Optional HuggingFace model ID. Defaults to ``ToxicityPrompts/PolyGuard-Ministral``.
    provider: Optional pre-configured provider. Defaults to a ``HuggingFaceProvider``
        loading a causal LM.

## Supported Models

- `ToxicityPrompts/PolyGuard-Ministral`
- `ToxicityPrompts/PolyGuard-Qwen`
- `ToxicityPrompts/PolyGuard-Qwen-Smol`

## Constructor

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `model_id` | `str | None` | No | `None` |
| `provider` | `Provider[dict[str, Any], dict[str, Any]] | None` | No | `None` |

Initialize the PolyGuard guardrail.

## validate

Classify ``input_text`` (and optionally an assistant ``output_text``).

**Parameters**

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `input_text` | `str` | Yes | — |
| `output_text` | `str | None` | No | `None` |

**Returns:** `GuardrailOutput`
