# Qwen3GuardStream

Qwen3Guard-Stream — token-level streaming safety moderation with span output (Qwen).

Classifier heads on a Qwen3 backbone (loaded as remote code) that judge the user
prompt as a whole and every assistant response token individually, each with a
three-level severity (``Safe`` / ``Controversial`` / ``Unsafe``, where
``Controversial`` means harmfulness is context-dependent). The model is multilingual
(the Qwen3Guard series covers up to 119 languages) and released under Apache-2.0.
``validate`` is a non-streaming facade over the token-level streaming API: it feeds
the whole prompt, then each ``output_text`` token in turn, and aggregates the worst
severity seen.

Verdict mapping onto ``GuardrailOutput``:

- ``valid`` is ``True`` only when everything judged is ``Safe``; when
  ``strict=False``, ``Controversial`` content also passes (only ``Unsafe`` fails).
- ``score`` maps the worst severity onto the canonical risk axis (higher = riskier):
  ``Safe`` → ``0.0``, ``Controversial`` → ``0.5``, ``Unsafe`` → ``1.0``.
- ``categories`` lists the distinct violation categories (``triggered=True``) seen
  across the prompt and non-``Safe`` response tokens.
- ``spans`` (response mode only) merges consecutive flagged response tokens into
  character-offset runs over ``output_text``, each labeled with its category and
  severity-derived ``score``; ``None`` when nothing is flagged or the tokenizer
  cannot supply offsets (a slow tokenizer degrades spans gracefully).
- ``extra`` carries the worst ``severity``, the ``prompt_severity``, and (in response
  mode) the ``response_severity``.
- Fails closed (``valid=False`` with ``extra={"parse_failure": True}``) when the
  backend reports no usable risk level.

For the generative variants (``Qwen3Guard-Gen-*``), see ``Qwen3Guard``.

Expected inputs: a single ``input_text`` (the user prompt; required) plus an optional
``output_text`` (the assistant response). With no ``output_text`` only the prompt is
moderated and no spans are produced. List/batch input is not supported — passing a
list raises ``TypeError``.

HuggingFace-only: the model ships its classification heads as remote code, so a
user-supplied provider must be a ``HuggingFaceProvider`` constructed with
``trust_remote_code=True``. The remote modeling code currently requires
``transformers>=4.51,<5`` (transformers 5 removed APIs it relies on); construction
raises ``ImportError`` on transformers >= 5.

For more information, see:

- [Qwen3Guard-Stream-0.6B model card](https://huggingface.co/Qwen/Qwen3Guard-Stream-0.6B) (default).
- [Qwen3Guard-Stream-4B model card](https://huggingface.co/Qwen/Qwen3Guard-Stream-4B).
- [Qwen3Guard-Stream-8B model card](https://huggingface.co/Qwen/Qwen3Guard-Stream-8B).
- [Qwen3Guard Technical Report (arXiv:2510.14276)](https://arxiv.org/abs/2510.14276)
- [Qwen3Guard: Real-time Safety for Your Token Stream (Qwen blog)](https://qwenlm.github.io/blog/qwen3guard/)

## Supported Models

- `Qwen/Qwen3Guard-Stream-0.6B`
- `Qwen/Qwen3Guard-Stream-4B`
- `Qwen/Qwen3Guard-Stream-8B`

## Constructor

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `strict` | `bool` | No | `True` | If ``True`` (default), only ``Safe`` verdicts pass validation; set ``False`` to let ``Controversial`` content pass (``valid=True``), leaving it reflected only in ``score``, ``extra``, and ``spans``. |
| `model_id` | `str | None` | No | `None` | Optional HuggingFace model ID. Must be one of ``SUPPORTED_MODELS``; defaults to ``Qwen/Qwen3Guard-Stream-0.6B``. |
| `provider` | `Optional[Provider[dict[str, Any], dict[str, Any]]]` | No | `None` | Optional pre-configured provider. Must be a ``HuggingFaceProvider`` constructed with ``trust_remote_code=True`` (the classification heads load as remote code); it is loaded with ``model_class=AutoModel`` / ``tokenizer_class=AutoTokenizer``. Defaults to one loading the remote-code model. |

Initialize the Qwen3GuardStream guardrail.

## validate

Moderate a user prompt and, optionally, the assistant response to it.

**Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `input_text` | `str` | Yes | — | The user prompt to moderate, e.g. ``"How do I make a bomb?"``. A single string; list/batch input is rejected with ``TypeError``. |
| `output_text` | `str | None` | No | `None` | Optional assistant response moderated token-by-token, e.g. ``"I can't help with that."``. When supplied, flagged token runs are returned as ``spans`` with character offsets into this text; when omitted, only the prompt is moderated and no spans are produced. |

**Returns:** `GuardrailOutput`
