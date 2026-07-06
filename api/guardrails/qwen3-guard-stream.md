# Qwen3GuardStream

Qwen3Guard-Stream — token-level streaming safety moderation (Apache-2.0).

Classifier heads on a Qwen3 backbone (loaded as remote code) that judge the user
prompt as a whole and every assistant response token individually, each with a
three-level severity (``Safe`` / ``Controversial`` / ``Unsafe``, where
``Controversial`` means harmfulness is context-dependent). ``validate`` is a
non-streaming facade: it feeds the full prompt, then each ``output_text`` token
through the streaming API and aggregates the worst severity. ``valid`` is ``True``
only when everything judged is ``Safe`` (``Controversial`` also passes when
``strict=False``); ``score`` maps the worst severity onto the canonical risk axis
(Safe 0.0, Controversial 0.5, Unsafe 1.0) and per-part severities are surfaced in
``extra``. In response mode, runs of flagged response tokens are returned as
``spans`` with character offsets into ``output_text``. Fails closed
(``valid=False`` with ``extra={"parse_failure": True}``) when the backend reports
no usable risk level. For the generative variants (``Qwen3Guard-Gen-*``), see
``Qwen3Guard``.

HuggingFace-only: the model ships its classification heads as remote code, so a
user-supplied provider must be a ``HuggingFaceProvider`` constructed with
``trust_remote_code=True``. The remote modeling code currently requires
``transformers>=4.51,<5`` (transformers 5 removed APIs it relies on); construction
raises ``ImportError`` on transformers >= 5.

For more information, see the model cards:

- [Qwen3Guard-Stream-0.6B](https://huggingface.co/Qwen/Qwen3Guard-Stream-0.6B) (default).
- [Qwen3Guard-Stream-4B](https://huggingface.co/Qwen/Qwen3Guard-Stream-4B).
- [Qwen3Guard-Stream-8B](https://huggingface.co/Qwen/Qwen3Guard-Stream-8B).

Args:
    strict: If ``True`` (default), only ``Safe`` verdicts pass validation; set
        ``False`` to let ``Controversial`` content pass (``valid=True``), leaving
        it reflected only in ``score``, ``extra``, and ``spans``.
    model_id: Optional HuggingFace model ID. Defaults to ``Qwen/Qwen3Guard-Stream-0.6B``.
    provider: Optional pre-configured ``HuggingFaceProvider`` with
        ``trust_remote_code=True``. Defaults to one loading the remote-code model.

## Supported Models

- `Qwen/Qwen3Guard-Stream-0.6B`
- `Qwen/Qwen3Guard-Stream-4B`
- `Qwen/Qwen3Guard-Stream-8B`

## Constructor

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `strict` | `bool` | No | `True` |
| `model_id` | `str | None` | No | `None` |
| `provider` | `Optional[Provider[dict[str, Any], dict[str, Any]]]` | No | `None` |

Initialize the Qwen3GuardStream guardrail.

## validate

Moderate ``input_text`` (or, when ``output_text`` is given, the assistant response to it).

**Parameters**

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `input_text` | `str` | Yes | — |
| `output_text` | `str | None` | No | `None` |

**Returns:** `GuardrailOutput`
