# KananaSafeguard

Kakao Kanana Safeguard — Korean safety guardrails.

Decoder LLMs that emit a single verdict token: ``<SAFE>`` or an ``<UNSAFE-*>`` code.
Three variants cover different taxonomies: harmful content (``-8b``, also judges an
assistant turn), legal/policy risk (``-siren-8b``), and prompt attacks (``-prompt-2.1b``).
``valid`` is ``True`` on ``<SAFE>``; the matched code is surfaced in ``categories``.
Fails closed (``valid=False`` with ``extra={"parse_failure": True}``) when no token parses.

For more information, see the model cards:

- [kanana-safeguard-8b](https://huggingface.co/kakaocorp/kanana-safeguard-8b) (default).
- [kanana-safeguard-siren-8b](https://huggingface.co/kakaocorp/kanana-safeguard-siren-8b).
- [kanana-safeguard-prompt-2.1b](https://huggingface.co/kakaocorp/kanana-safeguard-prompt-2.1b).

Args:
    model_id: Optional HuggingFace model ID. Defaults to ``kakaocorp/kanana-safeguard-8b``.
    provider: Optional pre-configured provider. Defaults to a ``HuggingFaceProvider``
        loading a causal LM.

## Supported Models

- `kakaocorp/kanana-safeguard-8b`
- `kakaocorp/kanana-safeguard-siren-8b`
- `kakaocorp/kanana-safeguard-prompt-2.1b`

## Constructor

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `model_id` | `str | None` | No | `None` |
| `provider` | `Optional[Provider[dict[str, Any], dict[str, Any]]]` | No | `None` |

Initialize the Kanana Safeguard guardrail.

## validate

Classify ``input_text`` (and, for the harm model, an assistant ``output_text``).

**Parameters**

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `input_text` | `str` | Yes | — |
| `output_text` | `str | None` | No | `None` |

**Returns:** `GuardrailOutput`
