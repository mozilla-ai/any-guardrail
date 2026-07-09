# Glider

GLIDER — prompt-based LLM judge that grades text against user-supplied pass criteria and rubric, returning reasoning and highlighted phrases (Patronus AI).

GLIDER is a compact (3B) evaluator LLM fine-tuned to score arbitrary text on
arbitrary user-defined criteria. Each call wraps the text in GLIDER's evaluation
prompt together with ``pass_criteria`` and ``rubric``; the model replies with a
``<reasoning>`` block, a ``<highlight>`` list of the decisive words/phrases, and
an integer ``<score>``.

Verdict mapping onto ``GuardrailOutput``:

- ``valid`` is ``rubric_score >= pass_threshold`` (or ``<=`` when
  ``higher_is_better=False``).
- ``score`` (canonical risk: higher = riskier) is populated only when
  ``score_range`` is supplied — the rubric score is normalized onto [0, 1] and
  inverted when higher rubric values mean better text. Otherwise ``score`` is
  ``None``.
- ``explanation`` is the model's ``<reasoning>`` block (the full generation when
  the block is missing).
- ``extra`` holds the raw integer ``rubric_score`` and the ``highlights`` string.
- When no ``<score>`` can be parsed, the output fails closed: ``valid=False``
  with ``extra={"parse_failure": True}``.

Inputs are single strings: ``input_text`` (required) plus an optional
``output_text`` judged alongside it (typically a model response); list/batch
input is not supported.

Note: this guardrail runs the model through a ``transformers`` text-generation
pipeline directly; the ``provider`` argument is reserved for future
extensibility and is currently unused.

For more information, see:

- [GLIDER model card](https://huggingface.co/PatronusAI/glider)
- [GLIDER: Grading LLM Interactions and Decisions using Explainable Ranking](https://arxiv.org/abs/2412.14140)


Raises:
    ValueError: Can only use model path to GLIDER from HuggingFace.

## Supported Models

- `PatronusAI/glider`

## Constructor

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `pass_criteria` | `str` | Yes | — | A question or description of what is being judged, e.g. ``"Is the response free of unsupported medical claims?"``. |
| `rubric` | `str` | Yes | — | A free-text scoring rubric telling the model what each score means, e.g. ``"0: contains unsupported claims. 1: all claims are supported."``. |
| `pass_threshold` | `int` | Yes | — | The rubric score at which the text counts as passing. ``valid`` is ``rubric_score >= pass_threshold`` (or ``<=`` when ``higher_is_better=False``). Must be on the same scale as the rubric. |
| `model_id` | `str | None` | No | `None` | Optional HuggingFace model ID. Must be one of ``SUPPORTED_MODELS``; defaults to ``PatronusAI/glider``. |
| `provider` | `Optional[Provider[dict[str, Any], dict[str, Any]]]` | No | `None` | Reserved for future extensibility; currently unused. GLIDER runs through a ``transformers`` text-generation pipeline instead. |
| `higher_is_better` | `bool` | No | `True` | Whether higher rubric scores mean better/passing text. Set to ``False`` for rubrics where higher scores mean worse text (e.g. a severity scale). |
| `score_range` | `tuple[int, int] | None` | No | `None` | Optional ``(min, max)`` bounds of the rubric scale, e.g. ``(0, 1)`` or ``(1, 5)``. Supplying it enables the normalized canonical risk in ``GuardrailOutput.score``; when omitted, ``score`` is ``None`` and the raw rubric value is still available in ``extra["rubric_score"]``. |

Initialize the GLIDER guardrail.

## validate

Use the provided pass criteria and rubric to judge the input and output text provided.

**Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `input_text` | `str` | Yes | — | The text to evaluate, wrapped in ``<INPUT>`` tags in GLIDER's evaluation prompt. Typically the user prompt or the standalone text being judged. Single string only; list/batch input is not supported. |
| `output_text` | `str | None` | No | `None` | Optional second text, wrapped in ``<OUTPUT>`` tags and judged alongside ``input_text`` — typically the model response when the pass criteria compare a response against a prompt (e.g. ``"Does the OUTPUT answer the question in the INPUT?"``). |

**Returns:** `GuardrailOutput`
