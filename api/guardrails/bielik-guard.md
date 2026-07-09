# BielikGuard

Bielik Guard — Polish multi-label safety classifier (SpeakLeash / Bielik.AI).

Encoder classifier that emits an independent probability (sigmoid) for each of five Polish
safety categories: Hate/Aggression, Vulgarities, Sexual Content, Crime, and Self-Harm. It
screens a single body of Polish text; category names are read from the model's ``id2label``
(the published card does not fix an index order), so the entries in ``categories`` follow
the model's own labels.

Verdict mapping onto ``GuardrailOutput``:

- ``categories`` carries every category with its sigmoid probability and a ``triggered``
  flag (probability strictly above ``threshold``).
- ``valid`` is ``True`` when no category exceeds ``threshold``.
- ``score`` (canonical risk: higher = riskier) is the maximum category probability.

Expected inputs: a single text string, or a ``list[str]`` for batched classification (the
inherited ``validate`` dispatches list input to ``_validate_batch``).

Two variants ship: the 0.1B default is a plain RoBERTa classifier; the 0.5B variant ships
custom modeling code and is loaded with ``trust_remote_code=True``. The repos are gated
(auto-approve) — authenticate with ``hf auth login`` before first use.

For more information, please see the model cards:

- [Bielik-Guard-0.1B-v1.1](https://huggingface.co/speakleash/Bielik-Guard-0.1B-v1.1) (default).
- [Bielik-Guard-0.5B-v1.1](https://huggingface.co/speakleash/Bielik-Guard-0.5B-v1.1).

## Supported Models

- `speakleash/Bielik-Guard-0.1B-v1.1`
- `speakleash/Bielik-Guard-0.5B-v1.1`

## Constructor

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model_id` | `str | None` | No | `None` | Optional HuggingFace model ID. Must be one of ``SUPPORTED_MODELS``; defaults to ``speakleash/Bielik-Guard-0.1B-v1.1``. Pass ``speakleash/Bielik-Guard-0.5B-v1.1`` for the larger variant (loaded with ``trust_remote_code=True`` because it ships custom modeling code). |
| `threshold` | `float` | No | `0.5` | Per-category probability strictly above which that category is flagged (and the text becomes invalid). Defaults to 0.5. |
| `provider` | `Optional[Provider[dict[str, Any], dict[str, Any]]]` | No | `None` | Optional pre-configured provider. If ``None``, a default ``HuggingFaceProvider`` is built with ``multi_label=True`` (and ``trust_remote_code`` enabled only for the 0.5B variant), then the model is loaded eagerly. |

Initialize the Bielik Guard guardrail.

## validate

Default validation pipeline: preprocess -> inference -> postprocess.

**Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `input_text` | `str | list[str]` | Yes | — | The text to validate. If a list is supplied, each item is validated and a list of GuardrailOutputs is returned in the same order. Subclasses can override ``_validate_batch`` to enable true batched inference; the default iterates over inputs. |

**Returns:** `GuardrailOutput | list[GuardrailOutput]`
