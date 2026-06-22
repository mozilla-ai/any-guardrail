# WildGuard

Allen Institute for AI WildGuard — one-stop safety moderation judge.

A generative classifier that evaluates, in a single pass: (1) whether the user
request is harmful, (2) whether the assistant response is a refusal, and
(3) whether the assistant response is harmful. ``valid`` is ``False`` when the
request is harmful or (when an ``output_text`` is supplied) the response is harmful.
The three signals are surfaced as ``categories``. Fails closed
(``valid=False`` with ``extra={"parse_failure": True}``) when no field parses.

For more information, see the
[WildGuard model card](https://huggingface.co/allenai/wildguard).

Args:
    model_id: Optional HuggingFace model ID. Defaults to ``allenai/wildguard``.
    provider: Optional pre-configured provider. Defaults to a ``HuggingFaceProvider``
        loading a causal LM.

## Supported Models

- `allenai/wildguard`

## Constructor

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `model_id` | `str | None` | No | `None` |
| `provider` | `Provider[dict[str, Any], dict[str, Any]] | None` | No | `None` |

Initialize the WildGuard guardrail.

## validate

Classify ``input_text`` (and optionally an assistant ``output_text``).

**Parameters**

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `input_text` | `str` | Yes | — |
| `output_text` | `str | None` | No | `None` |

**Returns:** `GuardrailOutput`
