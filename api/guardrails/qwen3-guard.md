# Qwen3Guard

Qwen3Guard-Gen — generative safety moderation with three-level severity (Apache-2.0).

Decoder LLM whose chat template embeds the safety-classifier instruction: the user
prompt alone triggers prompt moderation; supplying an assistant ``output_text``
switches to response moderation. The model reports a severity (``Safe`` /
``Controversial`` / ``Unsafe``, where ``Controversial`` means harmfulness is
context-dependent), the violated policy categories, and — in response mode —
whether the response is a refusal. ``valid`` is ``True`` only for ``Safe``
verdicts (``Controversial`` also passes when ``strict=False``); ``score`` maps
the severity onto the canonical risk axis (Safe 0.0, Controversial 0.5,
Unsafe 1.0) and the verbatim severity is surfaced in ``extra["severity"]``.
Fails closed (``valid=False`` with ``extra={"parse_failure": True}``) when no
severity parses. For the token-level streaming variants
(``Qwen3Guard-Stream-*``), see ``Qwen3GuardStream``.

For more information, see the model cards:

- [Qwen3Guard-Gen-0.6B](https://huggingface.co/Qwen/Qwen3Guard-Gen-0.6B) (default).
- [Qwen3Guard-Gen-4B](https://huggingface.co/Qwen/Qwen3Guard-Gen-4B).
- [Qwen3Guard-Gen-8B](https://huggingface.co/Qwen/Qwen3Guard-Gen-8B).

Args:
    strict: If ``True`` (default), only ``Safe`` verdicts pass validation; set
        ``False`` to let ``Controversial`` content pass (``valid=True``), leaving
        it reflected only in ``score`` and ``extra["severity"]``.
    model_id: Optional HuggingFace model ID. Defaults to ``Qwen/Qwen3Guard-Gen-0.6B``.
    provider: Optional pre-configured provider. Defaults to a ``HuggingFaceProvider``
        loading a causal LM.

## Supported Models

- `Qwen/Qwen3Guard-Gen-0.6B`
- `Qwen/Qwen3Guard-Gen-4B`
- `Qwen/Qwen3Guard-Gen-8B`

## Constructor

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `strict` | `bool` | No | `True` |
| `model_id` | `str | None` | No | `None` |
| `provider` | `Optional[Provider[dict[str, Any], dict[str, Any]]]` | No | `None` |

Initialize the Qwen3Guard guardrail.

## validate

Moderate ``input_text`` (or, when ``output_text`` is given, the assistant response to it).

**Parameters**

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `input_text` | `str` | Yes | — |
| `output_text` | `str | None` | No | `None` |

**Returns:** `GuardrailOutput`
