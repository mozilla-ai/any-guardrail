# OffTopic

Off-Topic — cross-encoder relevance detector that flags whether an input strays from a comparison text (GovTech Singapore).

A dispatcher over GovTech Singapore's two open off-topic detection models. Given an
``input_text`` and a ``comparison_text`` (typically the system prompt or the app's
intended topic), it decides whether the input is *on-topic* (relevant) or *off-topic*
(a distraction / topic drift). It selects the implementation from ``model_id``:

- ``mozilla-ai/jina-embeddings-v2-small-en-off-topic`` → ``OffTopicJina``, a
  bi-encoder that embeds the two texts separately (jina-embeddings-v2-small-en) and
  learns their relationship through cross-attention layers.
- ``mozilla-ai/stsb-roberta-base-off-topic`` (default) → ``OffTopicStsb``, a
  cross-encoder that concatenates the two texts and scores them jointly with a
  fine-tuned stsb-roberta-base.

Both are English-language models and truncate long inputs (Jina at 1024 tokens, STSB
at 514 tokens), emitting a ``warnings.warn`` when they do.

Verdict mapping onto ``GuardrailOutput``:

- ``valid`` is ``True`` when the input is on-topic, ``False`` when off-topic.
- ``score`` is ``P(off-topic)`` on the canonical risk axis (higher = riskier, i.e.
  more likely off-topic).
- ``categories`` reports both class probabilities: ``on-topic`` (score) and
  ``off-topic`` (score, with ``triggered=True`` when off-topic is the argmax class).

Expected inputs: two single strings. ``comparison_text`` is semantically required
even though it defaults to ``None`` — ``validate`` raises ``ValueError`` if it is
missing or empty. List/batch input is not supported.

For more information, see:

- [Off-Topic (STSB cross-encoder) model card](https://huggingface.co/mozilla-ai/stsb-roberta-base-off-topic) (default).
- [Off-Topic (Jina bi-encoder) model card](https://huggingface.co/mozilla-ai/jina-embeddings-v2-small-en-off-topic).
- [govtech/stsb-roberta-base-off-topic](https://huggingface.co/govtech/stsb-roberta-base-off-topic) (upstream).
- [govtech/jina-embeddings-v2-small-en-off-topic](https://huggingface.co/govtech/jina-embeddings-v2-small-en-off-topic) (upstream).
- [A Flexible Large Language Models Guardrail Development Methodology Applied to Off-Topic Prompt Detection (arXiv:2411.12946)](https://arxiv.org/abs/2411.12946)
- [Open-sourcing an Off-Topic Prompt Guardrail (GovTech blog)](https://medium.com/dsaid-govtech/open-sourcing-an-off-topic-prompt-guardrail-fde422a66152)

## Supported Models

- `mozilla-ai/jina-embeddings-v2-small-en-off-topic`
- `mozilla-ai/stsb-roberta-base-off-topic`

## Constructor

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model_id` | `str | None` | No | `None` | Optional HuggingFace model ID choosing which detector to load. Must be one of ``SUPPORTED_MODELS``: ``mozilla-ai/jina-embeddings-v2-small-en-off-topic`` loads the Jina bi-encoder (``OffTopicJina``); ``mozilla-ai/stsb-roberta-base-off-topic`` (the default) loads the STSB cross-encoder (``OffTopicStsb``). |
| `provider` | `Provider[dict[str, Any], dict[str, Any]] | None` | No | `None` | Reserved for future extensibility; currently unused. The selected implementation loads its model directly via ``transformers``. |

Initialize the Off-Topic guardrail, selecting the implementation from ``model_id``.

## validate

Judge whether ``input_text`` is on-topic relative to ``comparison_text``.

**Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `input_text` | `str` | Yes | — | The text to classify, e.g. a user prompt such as ``"What's the weather in Paris?"``. A single string. |
| `comparison_text` | `str | None` | No | `None` | The reference topic to compare against — typically the system prompt or the app's intended subject, e.g. ``"You are a customer-support bot for an online bookstore."``. Although it defaults to ``None`` for signature reasons, it is semantically required: a missing or empty value raises ``ValueError``. |

**Returns:** `GuardrailOutput`
