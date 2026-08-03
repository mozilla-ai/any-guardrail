# PolyGuard

Multilingual safety-moderation judge reporting request harm, response harm, and refusal across 17 languages.

Generative classifier (fine-tuned Ministral / Qwen decoder LLMs) that, given a human request
and an optional assistant response, reports three boolean signals — whether the request is
harmful, whether the response is a refusal, and whether the response is harmful — plus the
MLCommons hazard categories (``S1`` ... ``S14``) violated. Trained for moderation across 17
languages.

Verdict mapping onto ``GuardrailOutput``:

- ``valid`` is ``False`` when the request or the response is judged harmful.
- ``categories`` carries the three boolean signals (``harmful_request``, ``harmful_response``,
  ``response_refusal``) plus one ``CategoryResult`` per violated hazard code (``name`` = ``Sx``,
  ``description`` = the taxonomy label, ``triggered=True``), deduplicated in order of appearance.
- ``explanation`` is the raw generation.
- ``usage`` carries the prompt / completion token counts. No canonical ``score`` or ``spans``
  are produced.
- Fails closed (``valid=False`` with ``extra={"parse_failure": True}``) when neither
  harmfulness field parses.

Expected inputs: a single ``input_text`` string (the human request) plus an optional
``output_text`` string (the assistant response). The prompt template always carries both slots,
so ``output_text`` defaults to an empty response when omitted; supply it to have the response
judged for harm and refusal. Single strings only — passing a list raises ``TypeError``.

For more information, see the model cards:

- [ToxicityPrompts/PolyGuard-Ministral](https://huggingface.co/ToxicityPrompts/PolyGuard-Ministral) (default).
- [ToxicityPrompts/PolyGuard-Qwen](https://huggingface.co/ToxicityPrompts/PolyGuard-Qwen).
- [ToxicityPrompts/PolyGuard-Qwen-Smol](https://huggingface.co/ToxicityPrompts/PolyGuard-Qwen-Smol).

## Supported Models

- `ToxicityPrompts/PolyGuard-Ministral`
- `ToxicityPrompts/PolyGuard-Qwen`
- `ToxicityPrompts/PolyGuard-Qwen-Smol`

## Constructor

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model_id` | `str | None` | No | `None` | Optional HuggingFace model ID; must be one of ``SUPPORTED_MODELS``. Defaults to ``ToxicityPrompts/PolyGuard-Ministral``; ``ToxicityPrompts/PolyGuard-Qwen`` and ``ToxicityPrompts/PolyGuard-Qwen-Smol`` are the Qwen-based alternatives. |
| `provider` | `Optional[Provider[dict[str, Any], dict[str, Any]]]` | No | `None` | Optional pre-configured provider. When ``None``, a ``HuggingFaceProvider`` is built targeting a causal LM (``AutoModelForCausalLM`` + ``AutoTokenizer``). A supplied ``HuggingFaceProvider`` is corrected to those classes at load time; any other provider is used as-is. |
| `prompt` | `PromptTemplate | None` | No | `None` | Optional prompt-template override, used as-is (system prompt plus a user template filling ``{prompt}`` / ``{response}``). Defaults to ``None`` — the registry default, or the version named by ``prompt_version``. |
| `prompt_version` | `str | None` | No | `None` | Registered prompt version to use when ``prompt`` is not given. Defaults to ``None`` (the default version). See ``AnyGuardrail.list_prompt_versions``. |

Initialize the PolyGuard guardrail.

## validate

Classify ``input_text`` and, optionally, an assistant ``output_text``.

**Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `input_text` | `str` | Yes | — | The human request to moderate. Single string only. |
| `output_text` | `str | None` | No | `None` | Optional assistant response, judged for harm and refusal alongside the request. When omitted, the response slot is left empty. |

**Returns:** `GuardrailOutput`

## Benchmarks

### Content Safety

| Dataset (rev) | Metric | Threshold | Value | Harness | Source | Contam. |
| --- | --- | --- | --- | --- | --- | --- |
| openai_moderation (unspecified) | f1 | native-valid | 0.817276 | guardrail-bench+ag0.7.4 | measured:guardrail-bench+ag0.7.4 |  |
| xstest (unspecified) | fpr | native-valid | 0.016 | guardrail-bench+ag0.7.4 | measured:guardrail-bench+ag0.7.4 |  |
| wildguardmix (unspecified) | f1 | native-valid | 0.965753 | guardrail-bench+ag0.7.4 | measured:guardrail-bench+ag0.7.4 |  |
| aegis (unspecified) | f1 | native-valid | 0.855305 | guardrail-bench+ag0.7.4 | measured:guardrail-bench+ag0.7.4 |  |
| jbb (unspecified) | f1 | native-valid | 0.760456 | guardrail-bench+ag0.7.4 | measured:guardrail-bench+ag0.7.4 |  |
| orbench (unspecified) | fpr | native-valid | 0.733333 | guardrail-bench+ag0.7.4 | measured:guardrail-bench+ag0.7.4 |  |

## License

- **Vendor:** ToxicityPrompts
- **Default license:** `mrl` (of the default model/service)

| Model variant | License |
| --- | --- |
| `ToxicityPrompts/PolyGuard-Ministral` | `mrl` |
| `ToxicityPrompts/PolyGuard-Qwen` | `apache-2.0` |
| `ToxicityPrompts/PolyGuard-Qwen-Smol` | `apache-2.0` |
