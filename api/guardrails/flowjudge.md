# Flowjudge

Wrapper around FlowJudge, allowing for custom guardrailing based on user defined criteria, metrics, and rubric.

Please see the model card for more information: [FlowJudge](https://huggingface.co/flowaicom/Flow-Judge-v0.1).

Two ways to specify the evaluation. Either supply the convenience fields
(``name``/``criteria``/``rubric``/``required_inputs``/``required_output``) to
build a metric, or pass a prebuilt ``metric`` — a ``flow_judge`` ``Metric`` /
``CustomMetric`` or one of the library's preset metrics (e.g.
``RESPONSE_FAITHFULNESS_3POINT``).

Args:
    name: User defined metric name (convenience path).
    criteria: User defined question that they want answered by FlowJudge model (convenience path).
    rubric: A scoring rubric in a likert scale fashion, providing an integer score and then a description of what the
        value means (convenience path).
    required_inputs: A list of what is required for the judge to consider (convenience path).
    required_output: What is the expected output from the judge (convenience path).
    pass_threshold: The rubric score at which the text counts as passing. ``valid`` is
        ``rubric_score >= pass_threshold`` (or ``<=`` when ``higher_is_better`` is False).
    higher_is_better: Whether higher rubric scores mean better/passing text. Set to
        False for rubrics where higher scores mean worse text.
    metric: A prebuilt ``flow_judge`` ``Metric`` / ``CustomMetric`` (e.g. a preset). When
        given, the convenience fields are not required and are ignored.
    model: A prebuilt ``flow_judge`` backend (``Hf``, ``Vllm``, ``Llamafile``, ``Baseten``).
        Defaults to ``Hf(flash_attn=False)``. Use this to pick a faster/quantized backend
        (install the matching ``flow-judge[vllm|llamafile|baseten]`` extra yourself).
    generation_params: Generation parameters (``temperature``, ``top_p``, ``max_new_tokens``,
        ``do_sample``) for the default ``Hf`` backend. Ignored when ``model`` is supplied.

Raises:
    ImportError: When the ``flowjudge`` extra is not installed.
    ValueError: When neither ``metric`` nor the full convenience field set is provided.

## Constructor

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `name` | `str | None` | No | `None` |
| `criteria` | `str | None` | No | `None` |
| `rubric` | `dict[int, str] | None` | No | `None` |
| `required_inputs` | `list[str] | None` | No | `None` |
| `required_output` | `str | None` | No | `None` |
| `pass_threshold` | `int` | No | `3` |
| `higher_is_better` | `bool` | No | `True` |
| `metric` | `Any | None` | No | `None` |
| `model` | `Any | None` | No | `None` |
| `generation_params` | `dict[str, Any] | None` | No | `None` |

Initialize the FlowJudgeClass.

## validate

Classifies the desired input and output according to the associated metric provided to the judge.

**Parameters**

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `inputs` | `list[dict[str, str]]` | Yes | — |
| `output` | `dict[str, str]` | Yes | — |

**Returns:** `GuardrailOutput`
