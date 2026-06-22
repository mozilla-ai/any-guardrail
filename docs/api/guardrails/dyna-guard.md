# DynaGuard

DynaGuard — dynamic guardian model evaluating compliance with user-defined policies.

Decoder LLM that checks a transcript against a numbered list of natural-language
rules (the ``policy``) and returns ``PASS`` (compliant) or ``FAIL`` (a rule was
violated). ``valid`` is ``True`` on ``PASS``. With ``think=True`` the model emits
chain-of-thought reasoning before the verdict (stripped before parsing). Fails closed
(``valid=False`` with ``extra={"parse_failure": True}``) when no verdict parses.

For more information, see the model cards:

- [DynaGuard-8B](https://huggingface.co/tomg-group-umd/DynaGuard-8B) (default).
- [DynaGuard-4B](https://huggingface.co/tomg-group-umd/DynaGuard-4B).
- [DynaGuard-1.7B](https://huggingface.co/tomg-group-umd/DynaGuard-1.7B).

Args:
    policy: The rules to enforce, as numbered natural-language text.
    think: If ``True``, request chain-of-thought reasoning (higher latency).
    model_id: Optional HuggingFace model ID. Defaults to ``tomg-group-umd/DynaGuard-8B``.
    provider: Optional pre-configured provider. Defaults to a ``HuggingFaceProvider``
        loading a causal LM.

## Supported Models

- `tomg-group-umd/DynaGuard-8B`
- `tomg-group-umd/DynaGuard-4B`
- `tomg-group-umd/DynaGuard-1.7B`

## Constructor

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `policy` | `str` | Yes | — |
| `think` | `bool` | No | `False` |
| `model_id` | `str | None` | No | `None` |
| `provider` | `Provider[dict[str, Any], dict[str, Any]] | None` | No | `None` |

Initialize the DynaGuard guardrail.

## validate

Evaluate the transcript (``input_text`` plus optional agent ``output_text``) against the policy.

**Parameters**

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `input_text` | `str` | Yes | — |
| `output_text` | `str | None` | No | `None` |

**Returns:** `GuardrailOutput`
