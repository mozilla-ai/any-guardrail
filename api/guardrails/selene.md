# Selene

Atla Selene-1-Mini — general-purpose LLM judge.

Evaluates a response against a user-defined ``rubric`` on a 1-5 scale and returns
reasoning plus a score. ``valid`` maps the score through ``pass_threshold``;
``score`` is the rubric normalized onto the canonical risk axis; ``explanation`` is
the model's reasoning. Fails closed (``valid=False`` with
``extra={"parse_failure": True}``) when no score parses.

For more information, see the
[Selene-1-Mini model card](https://huggingface.co/AtlaAI/Selene-1-Mini-Llama-3.1-8B).

Args:
    rubric: The score rubric (objective plus ``Score 1:`` … ``Score 5:`` descriptions).
    pass_threshold: The score (1-5) at or above which the response passes.
    higher_is_better: Whether higher scores mean better. Defaults to ``True``.
    model_id: Optional HuggingFace model ID. Defaults to ``AtlaAI/Selene-1-Mini-Llama-3.1-8B``.
    provider: Optional pre-configured provider. Defaults to a ``HuggingFaceProvider``
        loading a causal LM.

## Supported Models

- `AtlaAI/Selene-1-Mini-Llama-3.1-8B`

## Constructor

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `rubric` | `str` | Yes | — |
| `pass_threshold` | `int` | Yes | — |
| `higher_is_better` | `bool` | No | `True` |
| `model_id` | `str | None` | No | `None` |
| `provider` | `Optional[Provider[dict[str, Any], dict[str, Any]]]` | No | `None` |

Initialize the Selene guardrail.

## validate

Judge ``output_text`` (the response) given ``input_text`` (the instruction).

**Parameters**

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `input_text` | `str` | Yes | — |
| `output_text` | `str | None` | No | `None` |

**Returns:** `GuardrailOutput`
