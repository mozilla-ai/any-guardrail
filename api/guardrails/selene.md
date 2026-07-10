# Selene

Selene 1 Mini — general-purpose LLM judge grading a response against a user-defined 1-5 rubric (Atla).

Selene 1 Mini is an 8B evaluator LLM (fine-tuned from Llama 3.1 8B) specialized in
scoring model outputs. This guardrail drives it in single-rubric absolute-grading mode:
each call wraps the instruction and response in Selene's evaluation prompt together with
the caller's ``rubric``, and the model replies with a ``**Reasoning:**`` block followed
by ``**Result:** <n>`` where ``n`` is an integer 1-5. It runs through
``provider.generate_chat``, so it can be served from either a ``HuggingFaceProvider`` or
a ``LlamafileProvider``.

Verdict mapping onto ``GuardrailOutput``:

- ``valid`` is ``rubric_score >= pass_threshold`` (or ``<=`` when
  ``higher_is_better=False``).
- ``score`` (canonical risk: higher = riskier) is the 1-5 rubric score normalized onto
  [0, 1] via ``normalize_rubric_to_risk`` — inverted when higher rubric values mean
  better, so a high-quality response yields a low risk.
- ``explanation`` is the model's full generation (reasoning plus the result line).
- ``extra["rubric_score"]`` is the raw integer 1-5.
- When no ``**Result:**`` score can be parsed, the output fails closed: ``valid=False``
  with ``extra={"parse_failure": True}``.

Inputs are single strings: ``input_text`` is the instruction and ``output_text`` is the
response being graded (when ``output_text`` is omitted, ``input_text`` is graded as the
response). List/batch input is not supported.

For more information, see:

- [Selene-1-Mini-Llama-3.1-8B model card](https://huggingface.co/AtlaAI/Selene-1-Mini-Llama-3.1-8B)
- [Atla Selene Mini: A General Purpose Evaluation Model (arXiv:2501.17195)](https://arxiv.org/abs/2501.17195)
- [Selene 1 Mini announcement (Atla)](https://www.atla-ai.com/post/selene-1-mini)

## Supported Models

- `AtlaAI/Selene-1-Mini-Llama-3.1-8B`

## Constructor

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `rubric` | `str` | Yes | — | The score rubric applied to every ``validate`` call — the evaluation objective plus ``Score 1:`` … ``Score 5:`` descriptions, e.g. ``"How helpful is the answer? Score 1: not helpful. ... Score 5: fully helpful."``. |
| `pass_threshold` | `int` | Yes | — | The score (1-5) at or above which the response passes (or at or below when ``higher_is_better=False``), e.g. ``4``. |
| `higher_is_better` | `bool` | No | `True` | Whether higher rubric scores mean better responses. Set ``False`` for rubrics where a higher number is worse. Defaults to ``True``. |
| `model_id` | `str | None` | No | `None` | Optional HuggingFace model ID; must be one of ``SUPPORTED_MODELS``. Defaults to ``AtlaAI/Selene-1-Mini-Llama-3.1-8B``. |
| `provider` | `Optional[Provider[dict[str, Any], dict[str, Any]]]` | No | `None` | Optional pre-configured provider. When ``None``, a ``HuggingFaceProvider`` is built targeting ``AutoModelForCausalLM`` / ``AutoTokenizer`` (transformers is imported lazily here). Pass a ``LlamafileProvider`` to run a GGUF build without the huggingface extra. |
| `prompt` | `PromptTemplate | None` | No | `None` | Optional prompt-template override, used as-is (must fill ``{instruction}`` / ``{response}`` / ``{rubric}``). Defaults to ``None`` — the registry default, or the version named by ``prompt_version``. |
| `prompt_version` | `str | None` | No | `None` | Registered prompt version to use when ``prompt`` is not given. Defaults to ``None`` (the default version). See ``AnyGuardrail.list_prompt_versions``. |

Initialize the Selene guardrail.

## validate

Judge ``output_text`` (the response) given ``input_text`` (the instruction).

**Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `input_text` | `str` | Yes | — | The instruction the response was produced for, e.g. ``"Summarize the article in one sentence."``. Single string only; list/batch input is not supported and raises ``TypeError``. |
| `output_text` | `str | None` | No | `None` | The response being graded against the rubric, e.g. ``"The article argues that remote work boosts productivity."``. When ``None``, ``input_text`` itself is graded as the response. |

**Returns:** `GuardrailOutput`
