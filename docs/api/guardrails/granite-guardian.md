# GraniteGuardian

Wrapper class for IBM Granite Guardian 4.1 models.

Granite Guardian is a hybrid-thinking safety/judge model that evaluates whether a
given text meets a user-specified criterion. It supports:

- **Bring-Your-Own-Criteria (BYOC)**: arbitrary natural-language criteria.
- **Predefined risks**: see `GraniteGuardianRisk` for strings covering safety,
  RAG hallucination, and function-calling hallucination.
- **RAG evaluation**: pass ``documents`` to `validate` to check groundedness,
  context relevance, or answer relevance.
- **Function-calling evaluation**: pass ``available_tools`` to `validate` to
  check for function-calling hallucinations.
- **Think / no-think modes**: set ``think=True`` to request chain-of-thought
  reasoning (higher latency, longer output).

The model returns ``yes`` when the text **meets** the criterion and ``no`` when
it does not. ``GuardrailOutput.valid`` follows the convention that criteria are
phrased as *violations* (e.g. ``"text contains harm"``), so ``valid`` is ``True``
when the model says ``no`` (safe) and ``False`` when it says ``yes`` (violation).
Phrase custom criteria accordingly.

For more information, see the
[IBM Granite Guardian model card](https://huggingface.co/ibm-granite/granite-guardian-4.1-8b).

Args:
    criteria: The judging criterion. Use a `GraniteGuardianRisk` constant or a
        custom string. Criteria should be phrased as violations for the default
        ``valid`` semantics to apply.
    think: If ``True``, run in think mode (chain-of-thought reasoning before
        scoring). Defaults to ``False`` for low-latency scoring.
    model_id: Optional HuggingFace model ID. Defaults to
        ``ibm-granite/granite-guardian-4.1-8b``.
    provider: Optional pre-configured provider. Defaults to a
        `HuggingFaceProvider` with ``AutoModelForCausalLM`` and ``AutoTokenizer``.

Raises:
    ValueError: If ``model_id`` is not in ``SUPPORTED_MODELS``.

## Supported Models

- `ibm-granite/granite-guardian-4.1-8b`

## Constructor

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `criteria` | `str` | Yes | — |
| `think` | `bool` | No | `False` |
| `model_id` | `str | None` | No | `None` |
| `provider` | `Provider[dict[str, Any], dict[str, Any]] | None` | No | `None` |

Initialize the Granite Guardian guardrail.

## validate

Score ``input_text`` (and optionally ``output_text``) against ``self.criteria``.

**Parameters**

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `input_text` | `str` | Yes | — |
| `output_text` | `str | None` | No | `None` |
| `documents` | `list[dict[str, Any]] | None` | No | `None` |
| `available_tools` | `list[dict[str, Any]] | None` | No | `None` |

**Returns:** `GuardrailOutput`
