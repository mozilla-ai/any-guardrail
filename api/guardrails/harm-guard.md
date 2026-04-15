# HarmGuard

Safety and jailbreak detection model based on DeBERTa-v3-large.

HarmAug-Guard classifies the safety of LLM conversations and detects jailbreak attempts.
It can evaluate either a single prompt or a prompt + response pair.

For more information, please see the model card:

- [HarmAug-Guard](https://huggingface.co/hbseong/HarmAug-Guard).

## Supported Models

- `hbseong/HarmAug-Guard`

## Constructor

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `model_id` | `str | None` | No | `None` |
| `threshold` | `float` | No | `0.5` |
| `provider` | `Optional[Provider[dict[str, Any], dict[str, Any]]]` | No | `None` |

Initialize the HarmGuard guardrail.

## validate

Validate whether the input (and optionally output) text is safe.

Args:
    input_text: The prompt/input text to evaluate.
    output_text: Optional response/output text. When provided, evaluates the
        safety of the response in context of the input.

Returns:
    GuardrailOutput with valid=True if safe, valid=False if harmful.
    The score represents the unsafe probability (0.0 = safe, 1.0 = unsafe).

**Parameters**

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `input_text` | `str` | Yes | — |
| `output_text` | `str | None` | No | `None` |

**Returns:** `GuardrailOutput[bool, NoneType, float]`
