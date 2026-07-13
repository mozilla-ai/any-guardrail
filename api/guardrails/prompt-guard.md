# PromptGuard

Encoder classifier for prompt-injection and jailbreak detection.

Binary encoder classifier (mDeBERTa / DeBERTa) that labels a single prompt string
``benign`` (index 0) or ``malicious`` (index 1, i.e. a prompt-injection or jailbreak
attempt). Meta's v2 collapsed Prompt Guard 1's separate "injection" class and focuses on
explicit jailbreak / injection attacks. It screens prompt text only — it is not designed
to judge model responses.

Verdict mapping onto ``GuardrailOutput``:

- ``valid`` is ``True`` when the predicted class is ``benign`` (argmax index != 1).
- ``score`` (canonical risk: higher = riskier) is the ``malicious`` probability.
- ``categories`` carries one ``CategoryResult`` per class, each with its probability and a
  ``triggered`` flag on the predicted class. The gated repos publish only the generic
  ``LABEL_0`` / ``LABEL_1`` names, so categories fall back to those when the provider
  surfaces no label list.

Expected inputs: a single prompt string, or a ``list[str]`` for batched classification
(the inherited ``validate`` dispatches list input to ``_validate_batch``). The 86M default
is multilingual; the 22M variant is English-only.

The repos are gated under the Llama 4 Community License — accept the terms on the model
page and authenticate with ``hf auth login`` before first use.

For more information, please see the model cards:

- [Llama-Prompt-Guard-2-86M](https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M)
  (default) — mDeBERTa-base, multilingual.
- [Llama-Prompt-Guard-2-22M](https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-22M)
  — DeBERTa-xsmall, English, ~75% lower latency.

## Supported Models

- `meta-llama/Llama-Prompt-Guard-2-86M`
- `meta-llama/Llama-Prompt-Guard-2-22M`

## Constructor

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model_id` | `str | None` | No | `None` | Optional HuggingFace model ID. Must be one of ``SUPPORTED_MODELS``; defaults to ``meta-llama/Llama-Prompt-Guard-2-86M`` (mDeBERTa-base, multilingual). Pass ``meta-llama/Llama-Prompt-Guard-2-22M`` for the smaller English-only variant with ~75% lower latency. |
| `provider` | `Optional[Provider[dict[str, Any], dict[str, Any]]]` | No | `None` | Optional pre-configured provider. If ``None``, a default ``HuggingFaceProvider`` (targeting ``AutoModelForSequenceClassification``) is built and the model is loaded eagerly. |

Initialize the Prompt Guard 2 guardrail.

## validate

Default validation pipeline: preprocess -> inference -> postprocess.

**Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `input_text` | `str | list[str]` | Yes | — | The text to validate. If a list is supplied, each item is validated and a list of GuardrailOutputs is returned in the same order. Subclasses can override ``_validate_batch`` to enable true batched inference; the default iterates over inputs. |

**Returns:** `GuardrailOutput | list[GuardrailOutput]`
