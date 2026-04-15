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

Args:
    endpoint (str): The endpoint URL for the Azure Content Safety service.
    api_key (str): The API key for authenticating with the service.
    threshold (int): The threshold for determining if content is unsafe.
    score_type (str): The type of score to use ("max" or "avg").
    blocklist_names (List[str] | None): List of blocklist names to use for content evaluation.

## validate

Validate content using Azure Content Safety.

Args:
    content (str): The content to be evaluated.

Returns:
    GuardrailOutput: The result of the guardrail evaluation.

**Parameters**

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `content` | `str` | Yes | — |

**Returns:** `GuardrailOutput[bool, dict[str, Union[int, list[str], NoneType]], float]`
