# KananaSafeguard

Korean safety decoder models covering harmful content, legal risk, and prompt attacks.

Decoder LLMs, trained primarily for Korean text, that emit a single verdict token:
``<SAFE>`` or an ``<UNSAFE-*>`` code. Three variants cover different taxonomies:

- ``kakaocorp/kanana-safeguard-8b`` (default): harmful content — Hate, Harassment,
  Sexual Content, Crime, Child Sexual Abuse, Self-Harm, Misinformation (``S1``-``S7``).
  The only variant trained to also judge an assistant turn (``output_text``).
- ``kakaocorp/kanana-safeguard-siren-8b``: legal/policy risk — Adult Authentication,
  Professional Advice, Personal Information, Intellectual Property (``I1``-``I4``).
- ``kakaocorp/kanana-safeguard-prompt-2.1b``: prompt attacks — Prompt Injection,
  Prompt Leaking (``A1``-``A2``).

Inputs are single strings (no batching): ``validate(input_text)`` for the user turn,
plus an optional assistant ``output_text`` that is only used by the harm (``-8b``)
model; the other variants ignore it.

``GuardrailOutput`` mapping: ``valid`` is ``True`` on ``<SAFE>``. On an ``<UNSAFE-*>``
verdict, ``categories`` holds one triggered entry named after the matched code (e.g.
``S3``) with its human-readable description, and ``extra["verdict"]`` carries the raw
token. ``score`` is not populated — the single-token verdict has no probability.
``explanation`` is the raw generated text. Fails closed (``valid=False`` with
``extra={"parse_failure": True}``) when no verdict token parses.

For more information, see:

- [kanana-safeguard-8b](https://huggingface.co/kakaocorp/kanana-safeguard-8b) (default).
- [kanana-safeguard-siren-8b](https://huggingface.co/kakaocorp/kanana-safeguard-siren-8b).
- [kanana-safeguard-prompt-2.1b](https://huggingface.co/kakaocorp/kanana-safeguard-prompt-2.1b).

## Supported Models

- `kakaocorp/kanana-safeguard-8b`
- `kakaocorp/kanana-safeguard-siren-8b`
- `kakaocorp/kanana-safeguard-prompt-2.1b`

## Constructor

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model_id` | `str | None` | No | `None` | Optional HuggingFace model ID; one of ``SUPPORTED_MODELS`` (``kakaocorp/kanana-safeguard-8b``, ``kakaocorp/kanana-safeguard-siren-8b``, ``kakaocorp/kanana-safeguard-prompt-2.1b``). Defaults to the harm model ``kakaocorp/kanana-safeguard-8b``. Each variant carries its own unsafe-code taxonomy (see the class docstring). |
| `provider` | `Optional[Provider[dict[str, Any], dict[str, Any]]]` | No | `None` | Optional pre-configured provider (e.g. a ``LlamafileProvider`` or a customized ``HuggingFaceProvider``). Defaults to a ``HuggingFaceProvider``; when a ``HuggingFaceProvider`` is used, the causal-LM loader classes (``AutoModelForCausalLM`` + ``AutoTokenizer``) are enforced at load time. |

Initialize the Kanana Safeguard guardrail.

## validate

Classify ``input_text`` (and, for the harm model, an assistant ``output_text``).

**Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `input_text` | `str` | Yes | — | The user turn to classify. Must be a single string — list (batch) inputs are not supported. Korean is the primary training language. |
| `output_text` | `str | None` | No | `None` | Optional assistant response. Only the harm model (``kakaocorp/kanana-safeguard-8b``) is trained to judge an assistant turn; the ``-siren-8b`` and ``-prompt-2.1b`` variants silently ignore this argument. |

**Returns:** `GuardrailOutput`
