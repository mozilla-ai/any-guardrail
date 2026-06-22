# GptOssSafeguard

OpenAI gpt-oss-safeguard — policy-grounded reasoning safety classifier.

A reasoning LLM that classifies content against a written ``policy`` supplied at
construction (bring-your-own-taxonomy). The policy becomes the system message; the
model reasons (OpenAI harmony format) and emits a verdict. A short output instruction
is appended so the reply ends with ``VIOLATION`` or ``SAFE``; ``valid`` is ``True`` on
``SAFE``. Fails closed (``valid=False`` with ``extra={"parse_failure": True}``) when no
verdict parses.

Note: the 120B variant is large; ``gpt-oss-safeguard-20b`` is the practical default.

For more information, see the model cards:

- [gpt-oss-safeguard-20b](https://huggingface.co/openai/gpt-oss-safeguard-20b) (default).
- [gpt-oss-safeguard-120b](https://huggingface.co/openai/gpt-oss-safeguard-120b).

Args:
    policy: The written safety policy the model evaluates content against.
    model_id: Optional HuggingFace model ID. Defaults to ``openai/gpt-oss-safeguard-20b``.
    provider: Optional pre-configured provider. Defaults to a ``HuggingFaceProvider``
        loading a causal LM.

## Supported Models

- `openai/gpt-oss-safeguard-20b`
- `openai/gpt-oss-safeguard-120b`

## Constructor

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `policy` | `str` | Yes | — |
| `model_id` | `str | None` | No | `None` |
| `provider` | `Provider[dict[str, Any], dict[str, Any]] | None` | No | `None` |

Initialize the gpt-oss-safeguard guardrail.

## validate

Classify ``input_text`` against the configured policy.

**Parameters**

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `input_text` | `str` | Yes | — |

**Returns:** `GuardrailOutput`
