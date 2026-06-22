# BielikGuard

Bielik Guard — Polish multi-label safety classifier (SpeakLeash / Bielik.AI).

Encoder classifier emitting independent per-category probabilities (sigmoid) over
five safety categories: Hate/Aggression, Vulgarities, Sexual Content, Crime, and
Self-Harm. ``valid`` is ``True`` when no category exceeds ``threshold``; ``score``
is the maximum category probability; ``categories`` carries every category. The
repos are gated (auto-approve) — authenticate with ``hf auth login`` first.

For more information, please see the model cards:

- [Bielik-Guard-0.1B-v1.1](https://huggingface.co/speakleash/Bielik-Guard-0.1B-v1.1) (default).
- [Bielik-Guard-0.5B-v1.1](https://huggingface.co/speakleash/Bielik-Guard-0.5B-v1.1).

Args:
    model_id: Optional HuggingFace model ID. Defaults to the 0.1B variant.
    threshold: Per-category probability above which a category is flagged. Defaults to 0.5.
    provider: Optional pre-configured provider. Defaults to a ``HuggingFaceProvider``
        with ``multi_label=True``.

## Supported Models

- `speakleash/Bielik-Guard-0.1B-v1.1`
- `speakleash/Bielik-Guard-0.5B-v1.1`

## Constructor

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `model_id` | `str | None` | No | `None` |
| `threshold` | `float` | No | `0.5` |
| `provider` | `Provider[dict[str, Any], dict[str, Any]] | None` | No | `None` |

Initialize the Bielik Guard guardrail.

## validate

Default validation pipeline: preprocess -> inference -> postprocess.

**Parameters**

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `input_text` | `str | list[str]` | Yes | — |

**Returns:** `GuardrailOutput | list[GuardrailOutput]`
