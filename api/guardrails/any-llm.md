# AnyLlm

A guardrail using `any-llm`.

## Constructor

Initialize self.  See help(type(self)) for accurate signature.

## validate

Validate the `input_text` against the given `policy`.

**Parameters**

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `input_text` | `str` | Yes | — |
| `policy` | `str` | Yes | — |
| `model_id` | `str` | No | `"openai:gpt-5-nano"` |
| `system_prompt` | `str` | No | `"You are a guardrail designed to ensure that the input text …"` |

**Returns:** `GuardrailOutput`
