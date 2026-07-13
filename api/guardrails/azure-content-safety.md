# AzureContentSafety

Hosted moderation of text and images across hate, sexual, self-harm, and violence categories with 0-7 severity scores.

Calls the Azure AI Content Safety ``analyze_text`` / ``analyze_image`` APIs. ``validate``
takes a single string: plain text is sent to text analysis, while a string that is an
existing local file path switches the guardrail to image-analysis mode and uploads that
image file. Text analysis can additionally match custom term blocklists, managed through
this class's blocklist helper methods (``create_or_update_blocklist``,
``add_blocklist_items``, etc.).

Verdict mapping: ``categories`` holds one entry per harm category (``hate``, ``self_harm``,
``sexual``, ``violence``) with Azure's raw ``severity`` (0-7 scale), a normalized ``score``
(severity / 7, higher = riskier), and ``triggered`` set when the severity reaches
``threshold``. The top-level ``score`` is the aggregate severity (``max`` or mean across
categories, per ``score_type``) normalized to [0, 1]. ``valid`` is ``True`` when the
aggregate severity is below ``threshold`` and no blocklist item matched; blocklist matches
are surfaced in ``extra["blocklists_match"]``.

Azure's moderation models are trained and tested on eight languages (Chinese, English,
French, German, Spanish, Italian, Japanese, Portuguese) and may work in others with
varying quality. Requires an Azure Content Safety resource: pass ``endpoint`` / ``api_key``
or set the ``CONTENT_SAFETY_ENDPOINT`` / ``CONTENT_SAFETY_KEY`` environment variables.

For more information, see:

- [Azure AI Content Safety overview](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/overview).
- [Harm categories and severity levels](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/concepts/harm-categories).
- [azure-ai-contentsafety Python SDK](https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/contentsafety/azure-ai-contentsafety).

## Supported Models

- `azure-content-safety`

## Constructor

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `endpoint` | `str | None` | No | `None` | The endpoint URL for the Azure Content Safety service. |
| `api_key` | `str | None` | No | `None` | The API key for authenticating with the service. |
| `threshold` | `int` | No | `2` | The threshold for determining if content is unsafe. |
| `score_type` | `str` | No | `"max"` | The type of score to use ("max" or "avg"). |
| `blocklist_names` | `list[str] | None` | No | `None` | List of blocklist names to use for content evaluation. |

Initialize Azure Content Safety client.

## validate

Validate content using Azure Content Safety.

**Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `content` | `str` | Yes | — | The content to be evaluated. |

**Returns:** `GuardrailOutput`
