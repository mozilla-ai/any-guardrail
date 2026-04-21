# OffTopic

Abstract base class for the Off Topic models.

For more information about the implementations about either off topic model, please see the below model cards:

- [govtech/stsb-roberta-base-off-topic model](https://huggingface.co/govtech/stsb-roberta-base-off-topic).
- [govtech/jina-embeddings-v2-small-en-off-topic](https://huggingface.co/govtech/jina-embeddings-v2-small-en-off-topic).

## Supported Models

- `mozilla-ai/jina-embeddings-v2-small-en-off-topic`
- `mozilla-ai/stsb-roberta-base-off-topic`

## Constructor

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `model_id` | `str | None` | No | `None` |
| `provider` | `Optional[Provider[dict[str, Any], dict[str, Any]]]` | No | `None` |

Off Topic model based on one of two implementations decided by model ID.

## validate

Compare two texts to see if they are relevant to each other.

**Parameters**

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `input_text` | `str` | Yes | — |
| `comparison_text` | `str | None` | No | `None` |

**Returns:** `GuardrailOutput[bool, dict[str, float], float]`
