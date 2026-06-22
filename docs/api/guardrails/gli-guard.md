# GliGuard

GLiGuard (Fastino) — schema-driven safety / toxicity / jailbreak / refusal detector.

Wraps the ``gliner2`` library to run a 300M encoder that classifies a text across
four tasks: prompt safety (safe/unsafe), prompt toxicity (15 categories),
jailbreak detection (12 attack types), and response refusal. ``valid`` is ``True``
when prompt safety is not ``unsafe`` and no jailbreak category fires. Triggered
toxicity and jailbreak labels are surfaced in ``categories``.

For more information, see the
[gliguard-LLMGuardrails-300M model card](https://huggingface.co/fastino/gliguard-LLMGuardrails-300M).

Args:
    model_id: Optional HuggingFace model ID. Defaults to ``fastino/gliguard-LLMGuardrails-300M``.
    threshold: Classification threshold passed to ``classify_text``. Defaults to 0.5.

Raises:
    ImportError: When the ``gliner`` extra is not installed.

## Supported Models

- `fastino/gliguard-LLMGuardrails-300M`

## Constructor

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `model_id` | `str | None` | No | `None` |
| `threshold` | `float` | No | `0.5` |

Initialize the GLiGuard guardrail.

## validate

Classify ``input_text`` across the safety/toxicity/jailbreak/refusal schema.

**Parameters**

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `input_text` | `str` | Yes | — |

**Returns:** `GuardrailOutput`
