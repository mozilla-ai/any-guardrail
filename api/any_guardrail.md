# AnyGuardrail

Factory class for creating guardrail instances.

## create

Create a guardrail instance.

**Parameters**

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `guardrail_name` | `GuardrailName` | Yes | — |
| `provider` | `Provider[Any, Any] | None` | No | `None` |

**Returns:** `Guardrail[Any, Any, Any]`

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
