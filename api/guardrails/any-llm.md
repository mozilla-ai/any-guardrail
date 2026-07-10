# AnyLlm

AnyLlm — policy-based LLM judge that grades text against a natural-language policy using any LLM provider supported by any-llm.

Wraps ``any_llm.completion`` with structured output: the judge model receives your policy
inside a system prompt and must return a boolean verdict, an explanation, and a risk score.
Any provider/model reachable through any-llm can be used (default: ``openai:gpt-5-nano``);
the chosen model must support structured output (``response_format``), and the provider's
credentials (e.g. ``OPENAI_API_KEY``) must be configured as any-llm expects.

Verdict mapping: ``valid`` is the judge's verdict (``True`` = compliant with the policy);
``score`` is the judge-reported risk in ``[0, 1]`` (higher = more likely violating the
policy); ``explanation`` is the judge's rationale; ``raw`` holds the full ``ChatCompletion``.
When the LLM response cannot be parsed into the expected schema, the output fails closed:
``valid=False`` with ``extra={"parse_failure": True}``.

Expected inputs: a single string plus a natural-language ``policy``, both passed per call to
``validate`` — this guardrail is stateless and takes no constructor arguments.

For more information, see:

- [any-llm on GitHub](https://github.com/mozilla-ai/any-llm).

## Constructor

Initialize self.  See help(type(self)) for accurate signature.

## validate

Validate the `input_text` against the given `policy`.

**Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `input_text` | `str` | Yes | — | The text to validate (a single string; sent as the user message). |
| `policy` | `str` | Yes | — | Natural-language policy to validate against, e.g. ``"The text must not request or contain personal data."``. Substituted into the system prompt's ``{policy}`` placeholder. |
| `model_id` | `str` | No | `"openai:gpt-5-nano"` | The judge model in any-llm's ``provider:model`` format, e.g. ``"openai:gpt-5-nano"`` (default) or ``"mistral:mistral-small-latest"``. The model must support structured output. |
| `system_prompt` | `str | None` | No | `None` | Override the system prompt. Defaults to ``None``, which uses the registry default (:data:`DEFAULT_SYSTEM_PROMPT`) — or the version selected by ``prompt_version``. A supplied prompt should have a ``{policy}`` placeholder and instruct the model to return the ``valid`` / ``explanation`` / ``risk_score`` fields of :class:`GuardrailOutputAnyLLM`. |
| `prompt_version` | `str | None` | No | `None` | Registered prompt version to use when ``system_prompt`` is not given. Defaults to ``None`` (the default version). See :meth:`any_guardrail.AnyGuardrail.list_prompt_versions`. |

**Returns:** `GuardrailOutput`
