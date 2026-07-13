# Jasper

Binary prompt-injection classifiers.

Runs one of JasperLS's encoder classifiers over a single user prompt and reports whether the
text is a prompt-injection attempt. Each model is a two-class sequence classifier whose unsafe
class is labeled ``"INJECTION"``; the guardrail treats that class as the risky one. The default
``gelectra-base-injection`` is built on a German-language gELECTRA encoder, while
``deberta-v3-base-injection`` is built on an English DeBERTa-v3 encoder — pick the one that
matches your prompt language.

Expected input: prompt-only text. ``validate(input_text)`` accepts a single string, or a
``list[str]`` to classify a batch in one call; there is no prompt+response or chat-message mode.

Verdict mapping onto ``GuardrailOutput``:

- ``valid`` is ``True`` when the predicted class is not ``"INJECTION"`` (i.e. the text looks safe).
- ``score`` is the model's probability of the ``"INJECTION"`` class (canonical risk direction:
  higher = riskier).
- ``categories`` carries one ``CategoryResult`` per class label, each with its softmax ``score``
  and a ``triggered`` flag marking the argmax class.
- No ``spans`` or ``modified_text`` are produced.

For more information, see:

- [gelectra-base-injection](https://huggingface.co/JasperLS/gelectra-base-injection) (default)
- [deberta-v3-base-injection](https://huggingface.co/JasperLS/deberta-v3-base-injection)


Raises:
    ValueError: If ``model_id`` is not in ``SUPPORTED_MODELS``.

## Supported Models

- `JasperLS/gelectra-base-injection`
- `JasperLS/deberta-v3-base-injection`

## Constructor

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model_id` | `str | None` | No | `None` | Optional HuggingFace model ID. Must be one of ``SUPPORTED_MODELS``; defaults to ``JasperLS/gelectra-base-injection`` (a German-language gELECTRA encoder). Pass ``"JasperLS/deberta-v3-base-injection"`` for the English DeBERTa-v3 variant. |
| `provider` | `Optional[Provider[dict[str, Any], dict[str, Any]]]` | No | `None` | Execution backend that loads the model and runs inference. Defaults to a ``HuggingFaceProvider`` targeting ``AutoModelForSequenceClassification``. Supply your own to control device, dtype, or ``cache_dir``, or to run against a different backend. |

Initialize the Jasper guardrail.

## validate

Default validation pipeline: preprocess -> inference -> postprocess.

**Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `input_text` | `str | list[str]` | Yes | — | The text to validate. If a list is supplied, each item is validated and a list of GuardrailOutputs is returned in the same order. Subclasses can override ``_validate_batch`` to enable true batched inference; the default iterates over inputs. |

**Returns:** `GuardrailOutput | list[GuardrailOutput]`
