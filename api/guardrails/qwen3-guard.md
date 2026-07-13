# Qwen3Guard

Generative safety moderation with three-level severity across 119 languages.

Decoder LLM (Apache-2.0) whose chat template embeds the safety-classifier
instruction: the user prompt alone triggers prompt moderation; supplying an
assistant ``output_text`` switches to response moderation. The model reports a
severity (``Safe`` / ``Controversial`` / ``Unsafe``, where ``Controversial``
means harmfulness is context-dependent), the violated categories from a
nine-item taxonomy (Violent, Non-violent Illegal Acts, Sexual Content or
Sexual Acts, PII, Suicide & Self-Harm, Unethical Acts, Politically Sensitive
Topics, Copyright Violation, plus Jailbreak for prompt moderation), and — in
response mode — whether the response is a refusal.

``GuardrailOutput`` mapping: ``valid`` is ``True`` only for ``Safe`` verdicts
(``Controversial`` also passes when ``strict=False``); ``score`` maps the
severity onto the canonical risk axis, higher = riskier (Safe 0.0,
Controversial 0.5, Unsafe 1.0); ``categories`` holds one triggered entry per
reported category (plus a ``refusal`` entry in response mode);
``extra["severity"]`` carries the verbatim severity and ``explanation`` the
full generation. Fails closed (``valid=False`` with
``extra={"parse_failure": True}``) when no severity parses. For the
token-level streaming variants (``Qwen3Guard-Stream-*``), see
``Qwen3GuardStream``.

For more information, see:

- [Qwen3Guard-Gen-0.6B model card](https://huggingface.co/Qwen/Qwen3Guard-Gen-0.6B) (default).
- [Qwen3Guard-Gen-4B model card](https://huggingface.co/Qwen/Qwen3Guard-Gen-4B).
- [Qwen3Guard-Gen-8B model card](https://huggingface.co/Qwen/Qwen3Guard-Gen-8B).
- [Qwen3Guard Technical Report](https://arxiv.org/abs/2510.14276).

## Supported Models

- `Qwen/Qwen3Guard-Gen-0.6B`
- `Qwen/Qwen3Guard-Gen-4B`
- `Qwen/Qwen3Guard-Gen-8B`

## Constructor

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `strict` | `bool` | No | `True` | If ``True`` (default), only ``Safe`` verdicts pass validation; set ``False`` to let ``Controversial`` content pass (``valid=True``), leaving it reflected only in ``score`` and ``extra["severity"]``. |
| `model_id` | `str | None` | No | `None` | Optional HuggingFace model ID, one of ``SUPPORTED_MODELS``. Defaults to ``Qwen/Qwen3Guard-Gen-0.6B``. |
| `provider` | `Optional[Provider[dict[str, Any], dict[str, Any]]]` | No | `None` | Optional pre-configured provider. Defaults to a ``HuggingFaceProvider`` loading a causal LM. |

Initialize the Qwen3Guard guardrail.

## validate

Moderate ``input_text`` (or, when ``output_text`` is given, the assistant response to it).

**Parameters**

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `input_text` | `str` | Yes | — |
| `output_text` | `str | None` | No | `None` |

**Returns:** `GuardrailOutput`
