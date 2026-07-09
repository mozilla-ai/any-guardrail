# Protectai

ProtectAI — binary prompt-injection classifiers built on DeBERTa-v3 and DistilRoBERTa (ProtectAI).

Runs one of ProtectAI's encoder classifiers over a single user prompt and reports whether the
text is a prompt-injection / jailbreak attempt. Each model is a two-class sequence classifier
whose unsafe class is labeled ``"INJECTION"``; the guardrail treats that class as the risky one.
``SUPPORTED_MODELS`` spans ProtectAI's LLM-security collection — three DeBERTa-v3 prompt-injection
models (small, base v1, base v2) plus a DistilRoBERTa "rejection" detector — with
``deberta-v3-small-prompt-injection-v2`` as the default.

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

- [ProtectAI LLM-security collection](https://huggingface.co/collections/protectai/llm-security-65c1f17a11c4251eeab53f40)
- [deberta-v3-small-prompt-injection-v2](https://huggingface.co/ProtectAI/deberta-v3-small-prompt-injection-v2) (default)
- [distilroberta-base-rejection-v1](https://huggingface.co/ProtectAI/distilroberta-base-rejection-v1)
- [deberta-v3-base-prompt-injection](https://huggingface.co/ProtectAI/deberta-v3-base-prompt-injection)
- [deberta-v3-base-prompt-injection-v2](https://huggingface.co/ProtectAI/deberta-v3-base-prompt-injection-v2)

## Supported Models

- `ProtectAI/deberta-v3-small-prompt-injection-v2`
- `ProtectAI/distilroberta-base-rejection-v1`
- `ProtectAI/deberta-v3-base-prompt-injection`
- `ProtectAI/deberta-v3-base-prompt-injection-v2`

## Constructor

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model_id` | `str | None` | No | `None` | Optional HuggingFace model ID. Must be one of ``SUPPORTED_MODELS``; defaults to ``ProtectAI/deberta-v3-small-prompt-injection-v2``. Pass one of the other entries (e.g. ``"ProtectAI/deberta-v3-base-prompt-injection-v2"``) for a larger DeBERTa-v3 classifier, or ``"ProtectAI/distilroberta-base-rejection-v1"`` for the rejection head. |
| `provider` | `Provider[dict[str, Any], dict[str, Any]] | None` | No | `None` | Execution backend that loads the model and runs inference. Defaults to a ``HuggingFaceProvider`` targeting ``AutoModelForSequenceClassification``. Supply your own to control device, dtype, or ``cache_dir``, or to run against a different backend. |

Initialize the ProtectAI guardrail.

## validate

Default validation pipeline: preprocess -> inference -> postprocess.

**Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `input_text` | `str | list[str]` | Yes | — | The text to validate. If a list is supplied, each item is validated and a list of GuardrailOutputs is returned in the same order. Subclasses can override ``_validate_batch`` to enable true batched inference; the default iterates over inputs. |

**Returns:** `GuardrailOutput | list[GuardrailOutput]`
