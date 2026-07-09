# Prometheus

Prometheus — open rubric-based LLM judge grading a response on a user-defined 1-5 rubric (KAIST / prometheus-eval).

Prometheus is an open-source decoder LLM specialized in evaluating other models'
outputs. This guardrail drives it in **absolute grading** mode: each call wraps the
instruction and response in Prometheus's evaluation prompt together with the caller's
``rubric`` (and an optional ``reference_answer`` that would score 5), and the model
replies with written feedback followed by ``[RESULT] <n>`` where ``n`` is an integer
1-5. It runs through ``provider.generate_chat``, so it can be served from either a
``HuggingFaceProvider`` or a ``LlamafileProvider``.

Verdict mapping onto ``GuardrailOutput``:

- ``valid`` is ``rubric_score >= pass_threshold`` (or ``<=`` when
  ``higher_is_better=False``).
- ``score`` (canonical risk: higher = riskier) is the 1-5 rubric score normalized onto
  [0, 1] via ``normalize_rubric_to_risk`` — inverted when higher rubric values mean
  better, so a high-quality response yields a low risk.
- ``explanation`` is the model's feedback (the text before the ``[RESULT]`` marker).
- ``extra["rubric_score"]`` is the raw integer 1-5.
- When no score can be parsed, the output fails closed: ``valid=False`` with
  ``extra={"parse_failure": True}``. The parser takes the **last** ``[RESULT]`` marker,
  because feedback often quotes other rubric levels inline.

Inputs are single strings: ``input_text`` is the instruction and ``output_text`` is the
response being graded (when ``output_text`` is omitted, ``input_text`` is graded as the
response). List/batch input is not supported.

For more information, see:

- [prometheus-7b-v2.0 model card](https://huggingface.co/prometheus-eval/prometheus-7b-v2.0) (default)
- [prometheus-8x7b-v2.0 model card](https://huggingface.co/prometheus-eval/prometheus-8x7b-v2.0)
- [prometheus-7b-v1.0 model card](https://huggingface.co/prometheus-eval/prometheus-7b-v1.0)
- [prometheus-13b-v1.0 model card](https://huggingface.co/prometheus-eval/prometheus-13b-v1.0)
- [Prometheus 2: An Open Source Language Model Specialized in Evaluating Other Language Models (arXiv:2405.01535)](https://arxiv.org/abs/2405.01535)
- [prometheus-eval/prometheus-eval on GitHub](https://github.com/prometheus-eval/prometheus-eval)

## Supported Models

- `prometheus-eval/prometheus-7b-v2.0`
- `prometheus-eval/prometheus-8x7b-v2.0`
- `prometheus-eval/prometheus-7b-v1.0`
- `prometheus-eval/prometheus-13b-v1.0`

## Constructor

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `rubric` | `str` | Yes | — | The score rubric applied to every ``validate`` call — the evaluation criteria plus ``Score 1:`` … ``Score 5:`` descriptions, e.g. ``"Is the answer factually correct? Score 1: entirely wrong. ... Score 5: fully correct."``. |
| `pass_threshold` | `int` | Yes | — | The score (1-5) at or above which the response passes (or at or below when ``higher_is_better=False``), e.g. ``4``. |
| `reference_answer` | `str | None` | No | `None` | Optional gold answer that would earn a score of 5, supplied to the model as an anchor for the top of the scale. Defaults to ``None`` (no reference; an empty string is sent). |
| `higher_is_better` | `bool` | No | `True` | Whether higher rubric scores mean better responses. Set ``False`` for rubrics where a higher number is worse (e.g. a severity scale). Defaults to ``True``. |
| `model_id` | `str | None` | No | `None` | Optional HuggingFace model ID; must be one of ``SUPPORTED_MODELS``. Defaults to ``prometheus-eval/prometheus-7b-v2.0``. |
| `provider` | `Provider[dict[str, Any], dict[str, Any]] | None` | No | `None` | Optional pre-configured provider. When ``None``, a ``HuggingFaceProvider`` is built targeting ``AutoModelForCausalLM`` / ``AutoTokenizer`` (transformers is imported lazily here). Pass a ``LlamafileProvider`` to run a GGUF build without the huggingface extra. |

Initialize the Prometheus guardrail.

## validate

Judge ``output_text`` (the response) given ``input_text`` (the instruction).

**Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `input_text` | `str` | Yes | — | The instruction the response was produced for, e.g. ``"Explain why the sky is blue to a five-year-old."``. Single string only; list/batch input is not supported and raises ``TypeError``. |
| `output_text` | `str | None` | No | `None` | The response being graded against the rubric, e.g. ``"The sky is blue because sunlight scatters off the air."``. When ``None``, ``input_text`` itself is graded as the response. |

**Returns:** `GuardrailOutput`
