# Glider

A prompt based guardrail from Patronus AI that utilizes pass criteria and a rubric to judge text.

For more information, see the model card:[GLIDER](https://huggingface.co/PatronusAI/glider). It outputs its reasoning,
highlights for what determined the score, and an integer score.

Args:
    pass_criteria: A question or description of what you are validating.
    rubric: A scoring rubric, describing to the model how to score the provided data.
    pass_threshold: The rubric score at which the text counts as passing. ``valid`` is
        ``rubric_score >= pass_threshold`` (or ``<=`` when ``higher_is_better`` is False).
    model_id: HuggingFace path to model.
    provider: Reserved for future extensibility. Currently unused.
    higher_is_better: Whether higher rubric scores mean better/passing text. Set to
        False for rubrics where higher scores mean worse text.

Raise:
    ValueError: Can only use model path to GLIDER from HuggingFace.

## Supported Models

- `PatronusAI/glider`

## Constructor

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `pass_criteria` | `str` | Yes | — |
| `rubric` | `str` | Yes | — |
| `pass_threshold` | `int` | Yes | — |
| `model_id` | `str | None` | No | `None` |
| `provider` | `Provider[dict[str, Any], dict[str, Any]] | None` | No | `None` |
| `higher_is_better` | `bool` | No | `True` |

Initialize the GLIDER guardrail.

## validate

Use the provided pass criteria and rubric to judge the input and output text provided.

**Parameters**

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `input_text` | `str` | Yes | — |
| `output_text` | `str | None` | No | `None` |

**Returns:** `GuardrailOutput`
