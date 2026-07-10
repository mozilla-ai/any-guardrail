# WildGuard

WildGuard — one-pass safety-moderation judge reporting prompt harm, response harm, and refusal (Allen Institute for AI).

WildGuard is a generative safety classifier that evaluates a prompt-response
interaction in a single forward pass, reporting three signals: (1) whether the
user request is harmful, (2) whether the assistant response is a refusal, and
(3) whether the assistant response is harmful. It is trained on the WildGuardMix
dataset and covers both vanilla (direct) prompts and adversarial jailbreaks.

Verdict mapping onto ``GuardrailOutput``:

- ``valid`` is ``False`` when the request is harmful, or — when an ``output_text``
  response is supplied — when the response is harmful; ``True`` otherwise.
- ``categories`` surfaces the three parsed signals as ``triggered`` booleans:
  ``harmful_request``, ``harmful_response``, and ``response_refusal``.
- ``explanation`` holds WildGuard's raw generation (the ``Harmful request: ... /
  Response refusal: ... / Harmful response: ...`` block).
- ``score`` is left ``None`` — WildGuard emits categorical yes/no verdicts rather
  than a calibrated risk probability.
- ``usage`` records the prompt/completion token counts when the backend reports them.
- Fails closed (``valid=False`` with ``extra={"parse_failure": True}``) when the
  always-present request verdict is missing, or when a response was being judged but
  its harm verdict could not be parsed (so a response is never silently passed as safe).

Expected inputs: a single ``input_text`` (the user request; required) plus an
optional ``output_text`` (the assistant response). With no ``output_text`` only the
request is judged and the response-side signals may be absent. List/batch input is
not supported — passing a list raises ``TypeError``.

Caveat: WildGuard ships its own instruction wrapper instead of a chat template, so
the prompt is fed to the model verbatim (``apply_chat_template=False``). That makes
it HuggingFace-only: ``LlamafileProvider`` rejects ``apply_chat_template=False``.

For more information, see:

- [WildGuard model card](https://huggingface.co/allenai/wildguard)
- [WildGuard: Open One-Stop Moderation Tools for Safety Risks, Jailbreaks, and Refusals of LLMs (arXiv:2406.18495)](https://arxiv.org/abs/2406.18495)

## Supported Models

- `allenai/wildguard`

## Constructor

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model_id` | `str | None` | No | `None` | Optional HuggingFace model ID. Must be one of ``SUPPORTED_MODELS``; defaults to ``allenai/wildguard``. |
| `provider` | `Optional[Provider[dict[str, Any], dict[str, Any]]]` | No | `None` | Optional pre-configured provider. Defaults to a ``HuggingFaceProvider`` loading the model as a causal LM. When a ``HuggingFaceProvider`` is supplied, it is loaded with ``model_class=AutoModelForCausalLM`` / ``tokenizer_class=AutoTokenizer`` so its default sequence-classification loader is corrected. |
| `prompt` | `PromptTemplate | None` | No | `None` | Optional prompt-template override, used as-is (must fill ``{prompt}`` / ``{response}``). Defaults to ``None`` — the registry default, or the version named by ``prompt_version``. |
| `prompt_version` | `str | None` | No | `None` | Registered prompt version to use when ``prompt`` is not given. Defaults to ``None`` (the default version). See ``AnyGuardrail.list_prompt_versions``. |

Initialize the WildGuard guardrail.

## validate

Classify a user request and, optionally, the assistant response to it.

**Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `input_text` | `str` | Yes | — | The user request to judge, e.g. ``"How do I pick a lock?"``. A single string; list/batch input is rejected with ``TypeError``. |
| `output_text` | `str | None` | No | `None` | Optional assistant response judged alongside the request, e.g. ``"I can't help with that."``. When omitted, only request harm is evaluated and the response-side signals may be absent. |

**Returns:** `GuardrailOutput`
