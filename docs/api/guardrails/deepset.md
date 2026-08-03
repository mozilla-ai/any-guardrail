# Deepset

Binary prompt-injection classifier.

Encoder sequence classifier that labels a text as ``LEGIT`` or ``INJECTION``.
Feed it the raw user prompt (or any untrusted text such as retrieved snippets);
no model response or extra context is involved. ``validate`` accepts a single
string or a ``list[str]``, in which case the whole list runs as one true
batched inference call and a list of outputs is returned in the same order.

Verdict mapping onto ``GuardrailOutput``:

- ``valid`` is ``True`` when the predicted label is not ``INJECTION``.
- ``score`` is the probability of the ``INJECTION`` class (canonical risk
  direction: higher = riskier), regardless of which label won the argmax.
- ``categories`` holds the full label distribution — one entry per class with
  its probability; the predicted class is marked ``triggered=True``.

For more information, see:

- [deepset/deberta-v3-base-injection model card](https://huggingface.co/deepset/deberta-v3-base-injection).

## Supported Models

- `deepset/deberta-v3-base-injection`

## Constructor

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `model_id` | `str | None` | No | `None` |
| `provider` | `Provider[dict[str, Any], dict[str, Any]] | None` | No | `None` |

Initialize the Deepset guardrail.

## validate

Default validation pipeline: preprocess -> inference -> postprocess.

**Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `input_text` | `str | list[str]` | Yes | — | The text to validate. If a list is supplied, each item is validated and a list of GuardrailOutputs is returned in the same order. Subclasses can override ``_validate_batch`` to enable true batched inference; the default iterates over inputs. |

**Returns:** `GuardrailOutput | list[GuardrailOutput]`

## Benchmarks

### Prompt Injection

| Dataset (rev) | Metric | Threshold | Value | Harness | Source | Contam. |
| --- | --- | --- | --- | --- | --- | --- |
| deepset_pi (unspecified) | f1 | native-valid | 0.991597 | guardrail-bench+ag0.7.4 | measured:guardrail-bench+ag0.7.4 | ⚠️ |
| notinject (unspecified) | fpr | native-valid | 0.705263 | guardrail-bench+ag0.7.4 | measured:guardrail-bench+ag0.7.4 |  |
| gandalf (unspecified) | recall | native-valid | 0.883929 | guardrail-bench+ag0.7.4 | measured:guardrail-bench+ag0.7.4 |  |
| bipia_email (unspecified) | f1 | native-valid | 0.7806 | bir@fd86c16 | measured:bir@fd86c16 |  |
| bipia_table (unspecified) | f1 | native-valid | 0.876783 | bir@fd86c16 | measured:bir@fd86c16 |  |

## License

- **Vendor:** deepset
- **Default license:** `mit` (of the default model/service)
