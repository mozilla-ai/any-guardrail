# OpenaiModeration

Guardrail implementation using OpenAI's Moderation API.

Wraps OpenAI's hosted moderation classifier (default model:
``omni-moderation-latest``) to flag content across 13 harm
sub-categories: hate, hate/threatening, harassment, harassment/threatening,
self-harm, self-harm/intent, self-harm/instructions, sexual, sexual/minors,
violence, violence/graphic, illicit, and illicit/violent.

The classifier returns a calibrated per-category probability score
alongside a boolean ``flagged`` verdict. This guardrail surfaces both:
``valid`` is ``False`` when OpenAI flags the content **or** when the
maximum per-category score exceeds ``threshold``; ``categories`` is the
full per-category breakdown (score + triggered flag); ``score`` is the
max category score.

The current ``omni-moderation`` model is a GPT-4o-derived multimodal
classifier; the original methodology (taxonomy, active-learning loop,
calibration) is described in Markov et al. 2023,
[A Holistic Approach to Undesired Content Detection in the Real World](https://arxiv.org/abs/2208.03274)
(AAAI 2023). See the
[omni-moderation announcement](https://openai.com/index/upgrading-the-moderation-api-with-our-new-multimodal-moderation-model/)
for the multimodal upgrade and the
[moderation guide](https://platform.openai.com/docs/guides/moderation)
for usage details. The Moderation API is free and does not count toward
standard usage quotas.

## Supported Models

- `omni-moderation-latest`
- `omni-moderation-2024-09-26`
- `text-moderation-latest`

## Constructor

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `model_id` | `str | None` | No | `None` |
| `api_key` | `str | None` | No | `None` |
| `base_url` | `str | None` | No | `None` |
| `threshold` | `float` | No | `0.5` |

Initialize the OpenAI Moderation guardrail.

## validate

Default validation pipeline: preprocess -> inference -> postprocess.

**Parameters**

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `input_text` | `str | list[str]` | Yes | — |

**Returns:** `GuardrailOutput | list[GuardrailOutput]`
