# Hhem

Vectara HHEM-2.1-Open — hallucination / factual-consistency detector.

A cross-encoder that scores how well a hypothesis (e.g. a generated answer) is
supported by a premise (e.g. the source document), returning a consistency score in
``[0, 1]`` where higher means better supported. ``validate(input_text, context=...)``
treats ``context`` as the premise and ``input_text`` as the hypothesis. ``valid`` is
``True`` when consistency ``>= threshold``; ``score`` is the canonical risk
(``1 - consistency``, higher = more likely hallucinated).

For more information, see the
[HHEM-2.1-Open model card](https://huggingface.co/vectara/hallucination_evaluation_model).

Args:
    threshold: Consistency at or above which the hypothesis is considered grounded.
    model_id: Optional HuggingFace model ID. Defaults to
        ``vectara/hallucination_evaluation_model``.

## Supported Models

- `vectara/hallucination_evaluation_model`

## Constructor

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `threshold` | `float` | No | `0.5` |
| `model_id` | `str | None` | No | `None` |

Initialize the HHEM guardrail.

## validate

Score how well ``input_text`` (hypothesis) is supported by ``context`` (premise).

**Parameters**

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `input_text` | `str` | Yes | — |
| `context` | `str | None` | No | `None` |

**Returns:** `GuardrailOutput`
