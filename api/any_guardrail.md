# AnyGuardrail

Factory class for creating guardrail instances.

## create

Create a guardrail instance.

**Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `guardrail_name` | `GuardrailName` | Yes | — | The name of the guardrail to use. |
| `provider` | `Optional[Provider[Any, Any]]` | No | `None` | Optional provider instance to use for model loading and inference. |

**Returns:** `Guardrail`

## metadata

Return the static taxonomy/capability metadata for a guardrail.

**Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `guardrail_name` | `GuardrailName` | Yes | — | The guardrail to look up. |

**Returns:** `GuardrailMetadata`

## list_guardrails

List guardrails whose metadata matches every supplied filter.

Filters combine with AND across dimensions. Set-valued dimensions
(``category``, ``stage``, ``output_shape``) accept a single enum member or
an iterable and match when the guardrail has **any** of the requested
values. Scalar dimensions match by equality. A ``None`` filter is ignored.
Runs entirely against the import-free registry — no backends are loaded.

**Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `category` | `GuardrailCategory | Iterable[GuardrailCategory] | None` | No | `None` | Keep guardrails detecting any of these categories. |
| `stage` | `GuardrailStage | Iterable[GuardrailStage] | None` | No | `None` | Keep guardrails that run at any of these stages. |
| `output_shape` | `OutputShape | Iterable[OutputShape] | None` | No | `None` | Keep guardrails producing any of these output shapes. |
| `backend` | `BackendType | None` | No | `None` | Keep guardrails with this backend. |
| `requires_api_key` | `bool | None` | No | `None` | Keep guardrails matching this API-key requirement. |
| `multilingual` | `bool | None` | No | `None` | Keep guardrails matching this multilingual flag. |
| `multimodal` | `bool | None` | No | `None` | Keep guardrails matching this multimodal flag. |
| `vendor` | `str | None` | No | `None` | Keep guardrails from this vendor. |

**Returns:** `list[GuardrailName]`

## group_by

Group guardrails by one metadata dimension.

**Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `dimension` | `str` | Yes | — | One of ``"category"``, ``"stage"``, ``"output_shape"``, ``"backend"``, or ``"vendor"``. For set-valued dimensions a guardrail appears under every value it carries. |

**Returns:** `dict[str, list[GuardrailName]]`

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
