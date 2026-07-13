# HarmGuard

Binary safety and jailbreak classifier, scoring a prompt or prompt-response pair.

HarmAug-Guard is a 435M DeBERTa-v3-large classifier distilled from a much larger (7B+)
teacher safety model using the HarmAug data-augmentation method, which jailbreaks an LLM
to synthesize harmful instructions for training. It classifies whether an LLM interaction
is safe or unsafe and flags jailbreak attempts, reaching an F1 comparable to 7B+ safety
models at a fraction of the compute.

Two input shapes are supported through ``validate``:

- a single prompt string — judges the prompt on its own; or
- a prompt plus a response (``output_text``) — tokenized as a text pair so the response is
  judged in the context of the prompt.

Verdict mapping onto ``GuardrailOutput``:

- ``score`` (canonical risk: higher = riskier) is the ``unsafe`` probability
  (``0.0`` = safe, ``1.0`` = unsafe), taken from ``softmax(logits)[1]`` — or from the
  column the provider labels ``unsafe`` when label names are available.
- ``valid`` is ``True`` when that unsafe probability is below ``threshold`` (default 0.5,
  from the paper).
- ``categories`` carries a ``safe`` and an ``unsafe`` entry with their probabilities and
  ``triggered`` flags.

For more information, see:

- [HarmAug-Guard model card](https://huggingface.co/hbseong/HarmAug-Guard).
- [HarmAug: Effective Data Augmentation for Knowledge Distillation of Safety Guard Models (arXiv:2410.01524)](https://arxiv.org/abs/2410.01524).

## Supported Models

- `hbseong/HarmAug-Guard`

## Constructor

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model_id` | `str | None` | No | `None` | Optional HuggingFace model ID. Must be one of ``SUPPORTED_MODELS``; defaults to ``hbseong/HarmAug-Guard``. |
| `threshold` | `float` | No | `0.5` | Unsafe-probability cutoff at or above which the input is flagged unsafe (``valid=False``). Defaults to 0.5, the value used in the HarmAug paper; lower it to catch borderline content, raise it to reduce false positives. |
| `provider` | `Optional[Provider[dict[str, Any], dict[str, Any]]]` | No | `None` | Optional pre-configured provider. If ``None``, a default ``HuggingFaceProvider`` is built and the model is loaded eagerly. |

Initialize the HarmGuard guardrail.

## validate

Validate whether the input (and optionally the response) is safe.

**Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `input_text` | `str` | Yes | — | The prompt / user text to evaluate, e.g. ``"How do I pick a lock?"``. A single string; list/batch input is not supported by this override. |
| `output_text` | `str | None` | No | `None` | Optional model response. When provided, ``input_text`` and ``output_text`` are tokenized as a text pair so the response is judged for safety in the context of the prompt; when ``None``, only the prompt is judged. |

**Returns:** `GuardrailOutput`
