# Sentinel

Binary prompt-injection classifier.

Runs Qualifire's DeBERTa-based encoder classifier over a single user prompt and reports whether
the text is a prompt-injection / jailbreak attempt. The model is a two-class sequence classifier
whose unsafe class is labeled ``"jailbreak"``; the guardrail treats that class as the risky one.

Expected input: prompt-only text. ``validate(input_text)`` accepts a single string, or a
``list[str]`` to classify a batch in one call; there is no prompt+response or chat-message mode.

Verdict mapping onto ``GuardrailOutput``:

- ``valid`` is ``True`` when the predicted class is not ``"jailbreak"`` (i.e. the text looks safe).
- ``score`` is the model's probability of the ``"jailbreak"`` class (canonical risk direction:
  higher = riskier).
- ``categories`` carries one ``CategoryResult`` per class label, each with its softmax ``score``
  and a ``triggered`` flag marking the argmax class.
- No ``spans`` or ``modified_text`` are produced.

For more information, see:

- [Sentinel model card](https://huggingface.co/qualifire/prompt-injection-sentinel)

## Supported Models

- `qualifire/prompt-injection-sentinel`

## Constructor

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model_id` | `str | None` | No | `None` | Optional HuggingFace model ID. Must be one of ``SUPPORTED_MODELS``; defaults to (and currently only supports) ``qualifire/prompt-injection-sentinel``. |
| `provider` | `Optional[Provider[dict[str, Any], dict[str, Any]]]` | No | `None` | Execution backend that loads the model and runs inference. Defaults to a ``HuggingFaceProvider`` targeting ``AutoModelForSequenceClassification``. Supply your own to control device, dtype, or ``cache_dir``, or to run against a different backend. |

Initialize the Sentinel guardrail.

## validate

Default validation pipeline: preprocess -> inference -> postprocess.

**Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `input_text` | `str | list[str]` | Yes | — | The text to validate. If a list is supplied, each item is validated and a list of GuardrailOutputs is returned in the same order. Subclasses can override ``_validate_batch`` to enable true batched inference; the default iterates over inputs. |

**Returns:** `GuardrailOutput | list[GuardrailOutput]`
