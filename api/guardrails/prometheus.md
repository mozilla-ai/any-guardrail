# Prometheus

Prometheus — open rubric-based LLM judge (KAIST).

Evaluates a response against a user-defined ``rubric`` on a 1-5 scale (absolute
grading) and returns feedback plus an integer score. ``valid`` maps the score
through ``pass_threshold``; ``score`` is the rubric normalized onto the canonical
risk axis; ``explanation`` is the model's feedback. Fails closed (``valid=False``
with ``extra={"parse_failure": True}``) when no score parses.

For more information, see the model cards:

- [prometheus-7b-v2.0](https://huggingface.co/prometheus-eval/prometheus-7b-v2.0) (default).
- [prometheus-8x7b-v2.0](https://huggingface.co/prometheus-eval/prometheus-8x7b-v2.0).
- [prometheus-7b-v1.0](https://huggingface.co/prometheus-eval/prometheus-7b-v1.0).
- [prometheus-13b-v1.0](https://huggingface.co/prometheus-eval/prometheus-13b-v1.0).

Args:
    rubric: The score rubric (criteria plus ``Score 1:`` … ``Score 5:`` descriptions).
    pass_threshold: The score (1-5) at or above which the response passes.
    reference_answer: Optional reference answer that would score 5.
    higher_is_better: Whether higher scores mean better. Defaults to ``True``.
    model_id: Optional HuggingFace model ID. Defaults to ``prometheus-eval/prometheus-7b-v2.0``.
    provider: Optional pre-configured provider. Defaults to a ``HuggingFaceProvider``
        loading a causal LM.

## Supported Models

- `prometheus-eval/prometheus-7b-v2.0`
- `prometheus-eval/prometheus-8x7b-v2.0`
- `prometheus-eval/prometheus-7b-v1.0`
- `prometheus-eval/prometheus-13b-v1.0`

## Constructor

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `rubric` | `str` | Yes | — |
| `pass_threshold` | `int` | Yes | — |
| `reference_answer` | `str | None` | No | `None` |
| `higher_is_better` | `bool` | No | `True` |
| `model_id` | `str | None` | No | `None` |
| `provider` | `Optional[Provider[dict[str, Any], dict[str, Any]]]` | No | `None` |

Initialize the Prometheus guardrail.

## validate

Judge ``output_text`` (the response) given ``input_text`` (the instruction).

**Parameters**

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `input_text` | `str` | Yes | — |
| `output_text` | `str | None` | No | `None` |

**Returns:** `GuardrailOutput`
