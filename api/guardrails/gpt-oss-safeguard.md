# GptOssSafeguard

Policy-grounded reasoning safety classifier that judges text against a bring-your-own written policy.

A reasoning LLM that classifies content against a written ``policy`` supplied at
construction (bring-your-own-taxonomy). The policy becomes the system message; the
model reasons (OpenAI harmony format) and emits a verdict. A short output instruction
is appended so the reply ends with ``VIOLATION`` or ``SAFE``.

Verdict mapping onto ``GuardrailOutput``:

- ``valid`` is ``True`` on ``SAFE`` and ``False`` on ``VIOLATION``.
- ``categories`` holds a single ``policy_violation`` entry (``triggered=True``
  on ``VIOLATION``).
- ``score`` is not populated — the model emits a discrete verdict, not a
  calibrated probability.
- ``extra["verdict"]`` carries the raw verdict string and ``explanation`` the
  full generation, including the model's reasoning.
- Fails closed (``valid=False`` with ``extra={"parse_failure": True}``) when no
  verdict parses.

Input is a single string ``input_text`` (the content to moderate, sent as the
user message); list/batch input is not supported, and there is no separate
response argument — include any conversation context in the text itself.

Note: the 120B variant is large; ``gpt-oss-safeguard-20b`` is the practical default.

For more information, see the model cards:

- [gpt-oss-safeguard-20b](https://huggingface.co/openai/gpt-oss-safeguard-20b) (default).
- [gpt-oss-safeguard-120b](https://huggingface.co/openai/gpt-oss-safeguard-120b).

## Supported Models

- `openai/gpt-oss-safeguard-20b`
- `openai/gpt-oss-safeguard-120b`

## Constructor

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `policy` | `str` | Yes | — | The written safety policy (bring-your-own-taxonomy) the model evaluates content against — e.g. a numbered list of disallowed-content rules with definitions and examples. It is installed as the system message, with a short output instruction appended so the model ends its reply with ``VIOLATION`` or ``SAFE``. |
| `model_id` | `str | None` | No | `None` | Optional HuggingFace model ID. Must be one of ``SUPPORTED_MODELS``: ``openai/gpt-oss-safeguard-20b`` (default) or ``openai/gpt-oss-safeguard-120b``. |
| `provider` | `Optional[Provider[dict[str, Any], dict[str, Any]]]` | No | `None` | Optional pre-configured provider (e.g. a ``LlamafileProvider`` or a customized ``HuggingFaceProvider``). Defaults to a ``HuggingFaceProvider`` loading the model with ``AutoModelForCausalLM``/``AutoTokenizer``; when a ``HuggingFaceProvider`` is supplied, the causal-LM loader classes are enforced at load time. |

Initialize the gpt-oss-safeguard guardrail.

## validate

Classify ``input_text`` against the configured policy.

**Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `input_text` | `str` | Yes | — | The content to moderate, sent as the user message beneath the policy system message. Single string only; list input raises ``TypeError``. |

**Returns:** `GuardrailOutput`
