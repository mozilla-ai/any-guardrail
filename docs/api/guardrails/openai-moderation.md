# OpenaiModeration

OpenAI Moderation — hosted moderation API flagging content across 13 harm categories with calibrated scores (OpenAI).

Wraps OpenAI's hosted moderation classifier (default model:
``omni-moderation-latest``) to flag content across 13 harm
sub-categories: hate, hate/threatening, harassment, harassment/threatening,
self-harm, self-harm/intent, self-harm/instructions, sexual, sexual/minors,
violence, violence/graphic, illicit, and illicit/violent. The classifier
returns a calibrated per-category probability score alongside a boolean
``flagged`` verdict.

Expected input: a single string (or a list of strings, screened one at a time
via the batched ``ThreeStageGuardrail.validate``).

``GuardrailOutput`` mapping:
    - ``valid`` is ``False`` when OpenAI flags the content **or** when the
      maximum per-category score exceeds ``threshold`` (otherwise ``True``).
    - ``score`` is the maximum per-category probability (higher = riskier).
    - ``categories`` is the full per-category breakdown: each
      ``CategoryResult`` carries the calibrated ``score`` and a ``triggered``
      flag (set when OpenAI flagged that category or its score exceeds
      ``threshold``).
    - ``raw`` is the full OpenAI SDK moderation response object.

The current ``omni-moderation`` model is a GPT-4o-derived multimodal
classifier; the original methodology (taxonomy, active-learning loop,
calibration) is described in Markov et al. 2023 (AAAI 2023). The Moderation
API is free and does not count toward standard usage quotas.

For more information, see:

- [Moderation guide (usage)](https://platform.openai.com/docs/guides/moderation)
- [Upgrading the Moderation API with our new multimodal moderation model](https://openai.com/index/upgrading-the-moderation-api-with-our-new-multimodal-moderation-model/)
- [A Holistic Approach to Undesired Content Detection in the Real World (arXiv:2208.03274)](https://arxiv.org/abs/2208.03274)

## Supported Models

- `omni-moderation-latest`
- `omni-moderation-2024-09-26`
- `text-moderation-latest`

## Constructor

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model_id` | `str | None` | No | `None` | The moderation model to use. Defaults to ``omni-moderation-latest``. |
| `api_key` | `str | None` | No | `None` | OpenAI API key. Falls back to the ``OPENAI_API_KEY`` environment variable when not provided. |
| `base_url` | `str | None` | No | `None` | Optional custom base URL for the OpenAI client (useful for proxies or Azure-style routing). |
| `threshold` | `float` | No | `0.5` | Maximum per-category score above which content is considered invalid even if OpenAI did not explicitly flag it. Defaults to ``0.5``. |

Initialize the OpenAI Moderation guardrail.

## validate

Default validation pipeline: preprocess -> inference -> postprocess.

**Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `input_text` | `str | list[str]` | Yes | — | The text to validate. If a list is supplied, each item is validated and a list of GuardrailOutputs is returned in the same order. Subclasses can override ``_validate_batch`` to enable true batched inference; the default iterates over inputs. |

**Returns:** `GuardrailOutput | list[GuardrailOutput]`
