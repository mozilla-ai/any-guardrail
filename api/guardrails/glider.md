# Glider

A prompt based guardrail from Patronus AI that utilizes pass criteria and a rubric to judge text.

For more information, see the model card:[GLIDER](https://huggingface.co/PatronusAI/glider). It outputs its reasoning,
highlights for what determined the score, and an integer score.

Args:
    model_id: HuggingFace path to model.
    pass_criteria: A question or description of what you are validating.
    rubric: A scoring rubric, describing to the model how to score the provided data.
    provider: Reserved for future extensibility. Currently unused.

Raise:
    ValueError: Can only use model path to GLIDER from HuggingFace.

## Supported Models

- `PatronusAI/glider`

## Constructor

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `pass_criteria` | `str` | Yes | — |
| `rubric` | `str` | Yes | — |
| `model_id` | `str | None` | No | `None` |
| `provider` | `Optional[Provider[dict[str, Any], dict[str, Any]]]` | No | `None` |

Initialize the GLIDER guardrail.

## validate

Use the provided pass criteria and rubric to judge the input and output text provided.

**Parameters**

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `input_text` | `str` | Yes | — |
| `output_text` | `str | None` | No | `None` |

**Returns:** `GuardrailOutput[NoneType, str, Union[int, NoneType]]`
