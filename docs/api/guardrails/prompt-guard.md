# PromptGuard

Meta Llama Prompt Guard 2 — encoder classifier for prompt injection / jailbreak detection.

Binary DeBERTa classifier that labels a prompt ``benign`` or ``malicious``.
``valid`` is ``True`` when the prompt is benign; ``score`` is the malicious
probability. The repos are gated under the Llama 4 Community License — accept
the terms and authenticate with ``hf auth login`` before first use.

For more information, please see the model cards:

- [Llama-Prompt-Guard-2-86M](https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M)
  (default) — mDeBERTa-base, multilingual.
- [Llama-Prompt-Guard-2-22M](https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-22M)
  — DeBERTa-xsmall, English, ~75% lower latency.

## Supported Models

- `meta-llama/Llama-Prompt-Guard-2-86M`
- `meta-llama/Llama-Prompt-Guard-2-22M`

## Constructor

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `model_id` | `str | None` | No | `None` |
| `provider` | `Provider[dict[str, Any], dict[str, Any]] | None` | No | `None` |

Initialize the Prompt Guard 2 guardrail.

## validate

Default validation pipeline: preprocess -> inference -> postprocess.

**Parameters**

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `input_text` | `str | list[str]` | Yes | — |

**Returns:** `GuardrailOutput | list[GuardrailOutput]`
