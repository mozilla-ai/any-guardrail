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
| `system_prompt` | `str` | No | `"
You are a guardrail designed to ensure that the input text adheres to a specific policy.
Your only task is to validate the input_text, don't try to answer the user query.

Here is the policy: {policy}

You must return the following:

- valid: bool
    If the input text provided by the user doesn't adhere to the policy, you must reject it (mark it as valid=False).

- explanation: str
    A clear explanation of why the input text was rejected or not.

- score: float (0-1)
    How confident you are about the validation.
"` |

**Returns:** `GuardrailOutput[bool, str, float]`
