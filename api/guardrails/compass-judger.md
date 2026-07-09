# CompassJudger

CompassJudger — generalist LLM judge that scores a response against user-defined criteria and rubric on a 1-10 scale (OpenCompass).

Decoder-LLM judge from the OpenCompass evaluation ecosystem. CompassJudger has no
canonical pointwise output format, so this guardrail wraps the inputs in a fixed
pointwise prompt instructing the model to give a brief justification and then emit
its verdict as ``Rating: [[X]]`` with an integer from 1 to 10. The verdict is the
*last* bracketed rating in the generation, so numbers the model quotes while
justifying are not mistaken for the final rating.

Inputs are single strings only (no batching): ``input_text`` is the instruction
and ``output_text`` is the response being judged. When ``output_text`` is omitted,
``input_text`` itself is placed in the response slot and judged directly.

Verdict mapping onto ``GuardrailOutput``:

- ``valid`` is ``True`` when the rating passes ``pass_threshold``
  (``rating >= pass_threshold`` when ``higher_is_better``, ``<=`` otherwise).
- ``score`` is the rating normalized onto the canonical risk axis in [0, 1]
  (higher = riskier), so a high rating under ``higher_is_better=True`` yields a
  low risk score.
- ``explanation`` is the model's full generated justification.
- ``extra["rubric_score"]`` is the raw 1-10 integer rating.
- Fails closed (``valid=False`` with ``extra={"parse_failure": True}``) when no
  in-range rating parses.

For more information, see:

- [CompassJudger-2-7B-Instruct](https://huggingface.co/opencompass/CompassJudger-2-7B-Instruct) (default).
- [CompassJudger-2-32B-Instruct](https://huggingface.co/opencompass/CompassJudger-2-32B-Instruct).
- [CompassJudger-1-1.5B-Instruct](https://huggingface.co/opencompass/CompassJudger-1-1.5B-Instruct).
- [CompassJudger-1-7B-Instruct](https://huggingface.co/opencompass/CompassJudger-1-7B-Instruct).
- [CompassJudger-1-14B-Instruct](https://huggingface.co/opencompass/CompassJudger-1-14B-Instruct).
- [CompassJudger-1-32B-Instruct](https://huggingface.co/opencompass/CompassJudger-1-32B-Instruct).
- [CompassJudger GitHub repository](https://github.com/open-compass/CompassJudger).
- [CompassJudger-2 paper (arXiv:2507.09104)](https://arxiv.org/abs/2507.09104).
- [CompassJudger-1 paper (arXiv:2410.16256)](https://arxiv.org/abs/2410.16256).

## Supported Models

- `opencompass/CompassJudger-2-7B-Instruct`
- `opencompass/CompassJudger-2-32B-Instruct`
- `opencompass/CompassJudger-1-1.5B-Instruct`
- `opencompass/CompassJudger-1-7B-Instruct`
- `opencompass/CompassJudger-1-14B-Instruct`
- `opencompass/CompassJudger-1-32B-Instruct`

## Constructor

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `criteria` | `str` | Yes | — | A description of what is being judged, e.g. ``"Helpfulness of the response to the user's question"``. |
| `rubric` | `str` | Yes | — | The scoring-rubric guidance describing what low vs. high ratings mean, e.g. ``"1-3: unhelpful ... 8-10: fully answers the question"``. |
| `pass_threshold` | `int` | Yes | — | The 1-10 rating at which the response passes. With ``higher_is_better=True``, ratings at or above it yield ``valid=True``; with ``higher_is_better=False``, ratings at or below it pass. |
| `higher_is_better` | `bool` | No | `True` | Whether higher ratings mean better text. Defaults to ``True``. |
| `model_id` | `str | None` | No | `None` | Optional HuggingFace model ID; must be one of ``SUPPORTED_MODELS``. Defaults to ``opencompass/CompassJudger-2-7B-Instruct``. |
| `provider` | `Optional[Provider[dict[str, Any], dict[str, Any]]]` | No | `None` | Optional pre-configured provider (e.g. a ``LlamafileProvider`` or a customized ``HuggingFaceProvider``). Defaults to a ``HuggingFaceProvider`` loading a causal LM. HuggingFace-backed loads force the SDPA attention kernel because CompassJudger-2's config requests flash_attention_2, which is unavailable on CPU/MPS. |

Initialize the CompassJudger guardrail.

## validate

Judge ``output_text`` (the response) given ``input_text`` (the instruction).

**Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `input_text` | `str` | Yes | — | The instruction/prompt the response answers, as a single string (list inputs are not supported), e.g. ``"Summarize the article in two sentences."``. |
| `output_text` | `str | None` | No | `None` | The response being judged — semantically the main text under evaluation. When ``None``, ``input_text`` itself is placed in the response slot of the judging prompt and judged directly. |

**Returns:** `GuardrailOutput`
