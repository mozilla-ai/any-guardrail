# CompassJudger

CompassJudger (OpenCompass) — generalist LLM judge.

Scores a response against a user-defined ``criteria`` and ``rubric`` on a 1-10 scale.
CompassJudger has no canonical pointwise output format, so this guardrail instructs it
to emit ``Rating: [[X]]``. ``valid`` maps the rating through ``pass_threshold``; ``score``
is normalized onto the canonical risk axis; ``explanation`` is the model's justification.
Fails closed (``valid=False`` with ``extra={"parse_failure": True}``) when no rating parses.

For more information, see the model cards:

- [CompassJudger-2-7B-Instruct](https://huggingface.co/opencompass/CompassJudger-2-7B-Instruct) (default).
- [CompassJudger-2-32B-Instruct](https://huggingface.co/opencompass/CompassJudger-2-32B-Instruct).
- [CompassJudger-1-1.5B-Instruct](https://huggingface.co/opencompass/CompassJudger-1-1.5B-Instruct).
- [CompassJudger-1-7B-Instruct](https://huggingface.co/opencompass/CompassJudger-1-7B-Instruct).
- [CompassJudger-1-14B-Instruct](https://huggingface.co/opencompass/CompassJudger-1-14B-Instruct).
- [CompassJudger-1-32B-Instruct](https://huggingface.co/opencompass/CompassJudger-1-32B-Instruct).

Args:
    criteria: A description of what is being judged.
    rubric: The scoring rubric guidance.
    pass_threshold: The rating (1-10) at or above which the response passes.
    higher_is_better: Whether higher ratings mean better. Defaults to ``True``.
    model_id: Optional HuggingFace model ID. Defaults to ``opencompass/CompassJudger-2-7B-Instruct``.
    provider: Optional pre-configured provider. Defaults to a ``HuggingFaceProvider``
        loading a causal LM.

## Supported Models

- `opencompass/CompassJudger-2-7B-Instruct`
- `opencompass/CompassJudger-2-32B-Instruct`
- `opencompass/CompassJudger-1-1.5B-Instruct`
- `opencompass/CompassJudger-1-7B-Instruct`
- `opencompass/CompassJudger-1-14B-Instruct`
- `opencompass/CompassJudger-1-32B-Instruct`

## Constructor

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `criteria` | `str` | Yes | — |
| `rubric` | `str` | Yes | — |
| `pass_threshold` | `int` | Yes | — |
| `higher_is_better` | `bool` | No | `True` |
| `model_id` | `str | None` | No | `None` |
| `provider` | `Provider[dict[str, Any], dict[str, Any]] | None` | No | `None` |

Initialize the CompassJudger guardrail.

## validate

Judge ``output_text`` (the response) given ``input_text`` (the instruction).

**Parameters**

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `input_text` | `str` | Yes | — |
| `output_text` | `str | None` | No | `None` |

**Returns:** `GuardrailOutput`
