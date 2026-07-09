# AnyGuardrail

Factory class for creating guardrail instances.

## create

Create a guardrail instance.

**Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `guardrail_name` | `GuardrailName` | Yes | — | The name of the guardrail to use. |
| `provider` | `Provider[Any, Any] | None` | No | `None` | Optional provider instance to use for model loading and inference. |

**Returns:** `Guardrail`

## get_supported_guardrails

List all supported guardrails.

**Returns:** `list[GuardrailName]`

## get_supported_model

Get the model IDs supported by a specific guardrail.

**Parameters**

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `guardrail_name` | `GuardrailName` | Yes | — |

**Returns:** `list[str]`

## get_all_supported_models

Get all model IDs supported by all guardrails.

**Returns:** `dict[str, list[str]]`
