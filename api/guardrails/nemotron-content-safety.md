# NemotronContentSafety

NVIDIA Nemotron Content Safety Reasoning — 4B safety classifier with optional reasoning.

Decoder LLM (Gemma-3-4B base) that classifies a prompt and optional response against
NVIDIA's 22-category content-safety taxonomy. ``valid`` is ``False`` when the prompt or
response is harmful. With ``think=True`` the model reasons inside ``<think>...</think>``
before the verdict (stripped before parsing). Fails closed (``valid=False`` with
``extra={"parse_failure": True}``) when no verdict parses. Distributed under the
NVIDIA Open Model License + Gemma Terms.

For more information, see the
[Nemotron-Content-Safety-Reasoning-4B model card](https://huggingface.co/nvidia/Nemotron-Content-Safety-Reasoning-4B).

Args:
    think: If ``True``, request chain-of-thought reasoning (``/think``); otherwise ``/no_think``.
    model_id: Optional HuggingFace model ID. Defaults to ``nvidia/Nemotron-Content-Safety-Reasoning-4B``.
    provider: Optional pre-configured provider. Defaults to a ``HuggingFaceProvider``
        loading a causal LM.

## Supported Models

- `nvidia/Nemotron-Content-Safety-Reasoning-4B`

## Constructor

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `think` | `bool` | No | `False` |
| `model_id` | `str | None` | No | `None` |
| `provider` | `Optional[Provider[dict[str, Any], dict[str, Any]]]` | No | `None` |

Initialize the Nemotron Content Safety guardrail.

## validate

Classify ``input_text`` (and optionally an assistant ``output_text``).

**Parameters**

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `input_text` | `str` | Yes | — |
| `output_text` | `str | None` | No | `None` |

**Returns:** `GuardrailOutput`
