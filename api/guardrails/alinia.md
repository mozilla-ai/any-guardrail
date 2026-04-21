# Alinia

Wraps the Alinia API for content moderation and safety detection.

This wrapper allows you to send conversations or text inputs to the Alinia API. You must get an API key from Alinia
and either set it to the ALINIA_API_KEY environment variable or pass it directly to the constructor. From Alinia, you'll also
be able to get the proper endpoint URL as well.

Args:
    endpoint (str): The Alinia API endpoint URL.
    detection_config (str | dict): The detection configuration ID or a dictionary specifying detection parameters.
    api_key (str | None): The API key for authenticating with the Alinia API. If not provided, it will be read from the ALINIA_API_KEY environment variable.
    metadata (dict | None): Optional metadata to include with the request.
    blocked_response (dict | None): Optional response to return if content is blocked.
    stream (bool): Whether to use streaming for the API response.

## Constructor

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `detection_config` | `str | dict[str, float | bool] | dict[str, dict[str, float | bool | str]]` | Yes | — |
| `api_key` | `str | None` | No | `None` |
| `endpoint` | `str | None` | No | `None` |
| `metadata` | `dict[str, Any] | None` | No | `None` |
| `blocked_response` | `dict[str, str] | None` | No | `None` |
| `stream` | `bool` | No | `False` |

Initialize the Alinia guardrail with the provided configuration.

## validate

Validate conversation or text input using the Alinia API.

This can be used for validation using any of the API endpoints provided by Alinia. If using sensitive information endpoint,
use the explanation from the GuardrailOutput to grab the recommended action text.

**Parameters**

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `conversation` | `str | list[dict[str, str]]` | Yes | — |
| `output` | `str | None` | No | `None` |
| `context_documents` | `list[str] | None` | No | `None` |

**Returns:** `GuardrailOutput[bool, dict[str, dict[str, Union[float, bool, str]]], dict[str, dict[str, float]]]`
