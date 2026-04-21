# AzureContentSafety

Guardrail implementation using Azure Content Safety service.

Azure Content Safety provides content moderation capabilities for text and images. To learn more about Azure
Content Safety, visit the [official documentation](https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/contentsafety/azure-ai-contentsafety`).

## Supported Models

- `azure-content-safety`

## Constructor

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `endpoint` | `str | None` | No | `None` |
| `api_key` | `str | None` | No | `None` |
| `threshold` | `int` | No | `2` |
| `score_type` | `str` | No | `"max"` |
| `blocklist_names` | `list[str] | None` | No | `None` |

Initialize Azure Content Safety client.

## validate

Validate content using Azure Content Safety.

**Parameters**

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `content` | `str` | Yes | — |

**Returns:** `GuardrailOutput[bool, dict[str, Union[int, list[str], NoneType]], float]`
