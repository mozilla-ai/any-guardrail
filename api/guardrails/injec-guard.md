# InjecGuard

PIGuard — binary prompt-injection classifier built on DeBERTa-v3 and trained to mitigate over-defense (successor to InjecGuard).

Runs PIGuard's DeBERTa-v3 encoder classifier over a single user prompt and reports whether the
text is a prompt-injection attempt. The model is a two-class sequence classifier whose unsafe
class is labeled ``"injection"``; the guardrail treats that class as the risky one. PIGuard adds
the "Mitigating Over-defense for Free" (MOF) training strategy (ACL 2025), which reduces the
trigger-word bias that makes prompt-injection guards falsely flag benign inputs.

Expected input: prompt-only text. ``validate(input_text)`` accepts a single string, or a
``list[str]`` to classify a batch in one call; there is no prompt+response or chat-message mode.

Verdict mapping onto ``GuardrailOutput``:

- ``valid`` is ``True`` when the predicted class is not ``"injection"`` (i.e. the text looks safe).
- ``score`` is the model's probability of the ``"injection"`` class (canonical risk direction:
  higher = riskier).
- ``categories`` carries one ``CategoryResult`` per class label, each with its softmax ``score``
  and a ``triggered`` flag marking the argmax class.
- No ``spans`` or ``modified_text`` are produced.

``leolee99/PIGuard`` is the renamed, maintained successor to InjecGuard (the rename was for
licensing reasons); ``leolee99/InjecGuard`` is the original repository, kept for backward
compatibility. Both share the same DeBERTa-v3 architecture and ``"injection"`` label and ship
custom model code, so the default provider loads them with ``trust_remote_code=True``.

For more information, see:

- [PIGuard model card](https://huggingface.co/leolee99/PIGuard) (default)
- [InjecGuard model card](https://huggingface.co/leolee99/InjecGuard)
- [InjecGuard: Benchmarking and Mitigating Over-defense in Prompt Injection Guardrail Models (arXiv:2410.22770)](https://arxiv.org/abs/2410.22770)

## Supported Models

- `leolee99/PIGuard`
- `leolee99/InjecGuard`

## Constructor

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model_id` | `str | None` | No | `None` | Optional HuggingFace model ID. Must be one of ``SUPPORTED_MODELS``; defaults to ``leolee99/PIGuard`` (the maintained successor). Pass ``"leolee99/InjecGuard"`` for the original repository, kept for backward compatibility. |
| `provider` | `Optional[Provider[dict[str, Any], dict[str, Any]]]` | No | `None` | Execution backend that loads the model and runs inference. Defaults to a ``HuggingFaceProvider`` constructed with ``trust_remote_code=True`` (PIGuard ships a custom model class), targeting ``AutoModelForSequenceClassification``. Supply your own to control device, dtype, or ``cache_dir``, or to run against a different backend. |

Initialize the PIGuard guardrail.

## validate

Default validation pipeline: preprocess -> inference -> postprocess.

**Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `input_text` | `str | list[str]` | Yes | — | The text to validate. If a list is supplied, each item is validated and a list of GuardrailOutputs is returned in the same order. Subclasses can override ``_validate_batch`` to enable true batched inference; the default iterates over inputs. |

**Returns:** `GuardrailOutput | list[GuardrailOutput]`
