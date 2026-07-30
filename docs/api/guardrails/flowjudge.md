# Flowjudge

Local LLM judge scoring text against user-defined criteria, metrics, and rubrics.

Flow Judge wraps Flow AI's ``flow-judge`` library and its 3.8B evaluator LLM
(``flowaicom/Flow-Judge-v0.1``, fine-tuned from Phi-3.5-mini). Unlike most guardrails it
bypasses the ``any-guardrail`` provider layer and drives ``flow_judge`` directly, so the
``flowjudge`` extra must be installed (a top-of-module ``ImportError`` is re-raised from
``__init__`` with a ``pip install`` hint otherwise).

Each guardrail is bound to a single ``flow_judge`` ``Metric`` (a criteria string plus a
Likert ``rubric``). There are two ways to specify it:

- **Convenience fields**: supply ``name`` / ``criteria`` / ``rubric`` /
  ``required_inputs`` / ``required_output`` and a ``Metric`` is built for you.
- **Prebuilt metric**: pass ``metric=`` — a ``flow_judge`` ``Metric`` / ``CustomMetric``
  or one of the library's presets (e.g. ``RESPONSE_FAITHFULNESS_3POINT``). The
  convenience fields are then ignored.

Verdict mapping onto ``GuardrailOutput``:

- ``valid`` is ``rubric_score >= pass_threshold`` (or ``<=`` when
  ``higher_is_better=False``).
- ``score`` (canonical risk: higher = riskier) is the Likert score normalized onto
  [0, 1] via ``normalize_rubric_to_risk``, using the rubric's integer keys as the
  scale bounds — inverted when higher rubric values mean better.
- ``explanation`` is the judge's feedback; ``extra["rubric_score"]`` is the raw integer.
- When the backend returns no score, the output fails closed: ``valid=False`` with
  ``extra={"parse_failure": True}``.

Inputs follow the ``flow_judge`` ``EvalInput`` shape rather than a plain string:
``validate(inputs, output)`` takes ``inputs`` — a list of single-key dicts, one per
``required_inputs`` name (e.g. ``[{"query": "..."}, {"context": "..."}]``) — and
``output``, a single-key dict for the ``required_output`` name (e.g.
``{"response": "..."}``). The keys must match the metric's declared inputs/output.

For more information, see:

- [Flow-Judge-v0.1 model card](https://huggingface.co/flowaicom/Flow-Judge-v0.1)
- [flowaicom/flow-judge on GitHub](https://github.com/flowaicom/flow-judge)
- [Flow Judge overview (Flow AI)](https://flow-ai.com/judge)


Raises:
    ImportError: When the ``flowjudge`` extra is not installed.
    ValueError: When neither ``metric`` nor the full convenience field set is provided.

## Constructor

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `name` | `str | None` | No | `None` | User-defined metric name (convenience path), e.g. ``"faithfulness"``. |
| `criteria` | `str | None` | No | `None` | User-defined question the judge should answer (convenience path), e.g. ``"Is the response grounded in the provided context?"``. |
| `rubric` | `dict[int, str] | None` | No | `None` | A Likert-scale ``dict[int, str]`` mapping each integer score to its meaning (convenience path). Its integer keys define the scale bounds used to normalize ``score``. |
| `required_inputs` | `list[str] | None` | No | `None` | The input field names the judge should consider (convenience path), e.g. ``["query", "context"]``; must match the ``inputs`` keys passed to ``validate``. |
| `required_output` | `str | None` | No | `None` | The output field name the judge should grade (convenience path), e.g. ``"response"``; must match the ``output`` key passed to ``validate``. |
| `pass_threshold` | `int` | No | `3` | The rubric score at which the response counts as passing. ``valid`` is ``rubric_score >= pass_threshold`` (or ``<=`` when ``higher_is_better`` is ``False``). Defaults to ``3``. |
| `higher_is_better` | `bool` | No | `True` | Whether higher rubric scores mean better responses. Set ``False`` for rubrics where a higher number is worse. Defaults to ``True``. |
| `metric` | `Any | None` | No | `None` | A prebuilt ``flow_judge`` ``Metric`` / ``CustomMetric`` or preset (e.g. ``RESPONSE_FAITHFULNESS_3POINT``). When given, the convenience fields are ignored. Keyword-only. |
| `model` | `Any | None` | No | `None` | A prebuilt ``flow_judge`` backend (``Hf``, ``Vllm``, ``Llamafile``, ``Baseten``). Defaults to ``Hf(flash_attn=False)``. Install the matching ``flow-judge[vllm\|llamafile\|baseten]`` extra yourself for non-default backends. Keyword-only. |
| `generation_params` | `dict[str, Any] | None` | No | `None` | Generation parameters (``temperature``, ``top_p``, ``max_new_tokens``, ``do_sample``) for the default ``Hf`` backend; ignored when ``model`` is supplied. Keyword-only. |

Initialize the Flow Judge guardrail.

Provide either a prebuilt ``metric=`` or the full set of convenience fields
(``name`` / ``criteria`` / ``rubric`` / ``required_inputs`` / ``required_output``);
the metric is built at construction time and the ``flow_judge`` backend is loaded.

## validate

Classifies the desired input and output according to the associated metric provided to the judge.

**Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `inputs` | `list[dict[str, str]]` | Yes | — | A list of single-key dictionaries, one per ``required_inputs`` name, each mapping that input name to its value, e.g. ``[{"query": "What is the capital of France?"}, {"context": "France's capital is Paris."}]``. |
| `output` | `dict[str, str]` | Yes | — | A single-key dictionary mapping the ``required_output`` name to the text being graded, e.g. ``{"response": "The capital of France is Paris."}``. |

**Returns:** `GuardrailOutput`
