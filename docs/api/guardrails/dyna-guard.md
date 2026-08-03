# DynaGuard

Dynamic guardian model evaluating conversation compliance with user-defined policies.

A decoder-LLM guardian model (University of Maryland / Capital One) that checks a
conversation transcript against a bring-your-own ``policy`` — a numbered list of
natural-language rules — and returns ``PASS`` (compliant) or ``FAIL`` (at least one
rule violated). Unlike fixed-taxonomy safety classifiers, the rules are
application-specific: e.g. "the agent must never issue a refund" or "the agent must
not give medical advice". With ``think=True`` the model first emits a
chain-of-thought ``<think>`` block justifying each rule before the ``<answer>``
verdict (higher latency, potentially higher accuracy); the reasoning is stripped
before the verdict is parsed.

Verdict mapping onto ``GuardrailOutput``:

- ``valid`` is ``True`` on ``PASS`` and ``False`` on ``FAIL``.
- ``categories`` carries a single ``policy_violation`` entry whose ``triggered``
  flag is ``True`` when the verdict is ``FAIL``.
- ``extra["verdict"]`` holds the raw ``"PASS"`` / ``"FAIL"`` token.
- ``explanation`` holds the model's full raw generation (including any reasoning).
- ``score`` is left ``None`` — DynaGuard emits a categorical verdict, not a
  calibrated risk probability.
- ``usage`` records the prompt/completion token counts when the backend reports them.
- Fails closed (``valid=False`` with ``extra={"parse_failure": True}``) when neither
  an ``<answer>`` block nor a bare ``PASS``/``FAIL`` token can be parsed (e.g. the
  generation was truncated mid-reasoning).

Expected inputs: ``input_text`` (the user turn / transcript; required) plus an
optional ``output_text`` (the agent's response). The two are assembled into a
``User: ... Agent: ...`` transcript (the turns joined by a newline) before being
wrapped with the ``policy``. ``input_text`` also accepts a ``list[str]`` to judge a
batch in one real batched ``generate_chat`` call when the provider is a
``HuggingFaceProvider`` (``output_text`` may then be a matching-length list, a
single value broadcast to every item, or omitted); other providers fall back to
one call per item.

For more information, see:

- [DynaGuard-8B model card](https://huggingface.co/tomg-group-umd/DynaGuard-8B) (default).
- [DynaGuard-4B model card](https://huggingface.co/tomg-group-umd/DynaGuard-4B).
- [DynaGuard-1.7B model card](https://huggingface.co/tomg-group-umd/DynaGuard-1.7B).
- [DynaGuard: A Dynamic Guardian Model With User-Defined Policies (arXiv:2509.02563)](https://arxiv.org/abs/2509.02563)

## Supported Models

- `tomg-group-umd/DynaGuard-8B`
- `tomg-group-umd/DynaGuard-4B`
- `tomg-group-umd/DynaGuard-1.7B`

## Constructor

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `policy` | `str` | Yes | — | The rules to enforce, as numbered natural-language text (a bring-your-own taxonomy), e.g. a newline-separated list of ``1. Do not reveal the system prompt.`` and ``2. Do not issue refunds.``. Applied to every ``validate`` call. |
| `think` | `bool` | No | `False` | If ``True``, request chain-of-thought reasoning before the verdict, which raises the generation token budget and latency. Defaults to ``False``. |
| `model_id` | `str | None` | No | `None` | Optional HuggingFace model ID. Must be one of ``SUPPORTED_MODELS``; defaults to ``tomg-group-umd/DynaGuard-8B``. |
| `provider` | `Provider[dict[str, Any], dict[str, Any]] | None` | No | `None` | Optional pre-configured provider. Defaults to a ``HuggingFaceProvider`` loading the model as a causal LM. When a ``HuggingFaceProvider`` is supplied, it is loaded with ``model_class=AutoModelForCausalLM`` / ``tokenizer_class=AutoTokenizer``. |
| `prompt` | `PromptTemplate | None` | No | `None` | Optional prompt-template override, used as-is (system prompt plus a user template filling ``{policy}`` / ``{transcript}``). Defaults to ``None`` — the registry default, or the version named by ``prompt_version``. |
| `prompt_version` | `str | None` | No | `None` | Registered prompt version to use when ``prompt`` is not given. Defaults to ``None`` (the default version). See ``AnyGuardrail.list_prompt_versions``. |

Initialize the DynaGuard guardrail.

## validate

Evaluate a conversation transcript against the configured policy.

**Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `input_text` | `str | list[str]` | Yes | — | The user turn (or the transcript to evaluate), e.g. ``"Please refund my last order."``, or a ``list[str]`` to judge a batch in one call. |
| `output_text` | `str | list[str] | None` | No | `None` | Optional agent response judged alongside the user turn, e.g. ``"Sure, I've issued your refund."``. When supplied, the two are assembled into a ``User: ... Agent: ...`` transcript (turns joined by a newline). For a batched ``input_text``, this may be a matching-length list, a single value broadcast to every item, or omitted. |

**Returns:** `GuardrailOutput | list[GuardrailOutput]`

## Benchmarks

### Content Safety

| Dataset (rev) | Metric | Threshold | Value | Harness | Source | Contam. |
| --- | --- | --- | --- | --- | --- | --- |
| openai_moderation (unspecified) | f1 | native-valid | 0.81746 | guardrail-bench+ag0.7.4 | measured:guardrail-bench+ag0.7.4 |  |
| xstest (unspecified) | fpr | native-valid | 0.032 | guardrail-bench+ag0.7.4 | measured:guardrail-bench+ag0.7.4 |  |
| wildguardmix (unspecified) | f1 | native-valid | 0.933824 | guardrail-bench+ag0.7.4 | measured:guardrail-bench+ag0.7.4 |  |
| aegis (unspecified) | f1 | native-valid | 0.744 | guardrail-bench+ag0.7.4 | measured:guardrail-bench+ag0.7.4 |  |
| jbb (unspecified) | f1 | native-valid | 0.834123 | guardrail-bench+ag0.7.4 | measured:guardrail-bench+ag0.7.4 |  |
| orbench (unspecified) | fpr | native-valid | 0.242105 | guardrail-bench+ag0.7.4 | measured:guardrail-bench+ag0.7.4 |  |

### General Judge

| Dataset (rev) | Metric | Threshold | Value | Harness | Source | Contam. |
| --- | --- | --- | --- | --- | --- | --- |
| judgebench (unspecified) | choice_accuracy | native-valid | 0.557895 | router_judge@c06603a | measured:router_judge@c06603a |  |
| llmbar (unspecified) | choice_accuracy | native-valid | 0.480702 | router_judge@c06603a | measured:router_judge@c06603a |  |
| rewardbench2 (unspecified) | choice_accuracy | native-valid | 0.2 | router_judge@c06603a | measured:router_judge@c06603a |  |

### Tool Use

| Dataset (rev) | Metric | Threshold | Value | Harness | Source | Contam. |
| --- | --- | --- | --- | --- | --- | --- |
| bfcl (unspecified) | accuracy | native-valid | 0.880702 | router_judge@c06603a | measured:router_judge@c06603a |  |
| hammerbench (unspecified) | accuracy | native-valid | 0.617544 | router_judge@c06603a | measured:router_judge@c06603a |  |

## License

- **Vendor:** DynaGuard
- **Default license:** `apache-2.0` (of the default model/service)
