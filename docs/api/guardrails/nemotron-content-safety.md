# NemotronContentSafety

Safety classifier covering NVIDIA's multi-category content-safety taxonomy.

Decoder LLM that classifies a user prompt and an optional assistant response against NVIDIA's
content-safety taxonomy. Two variants are supported, and they differ in base model, taxonomy,
and output format:

- ``nvidia/Nemotron-Content-Safety-Reasoning-4B`` (default, Gemma-3-4B base) — 22 categories
  (``S1`` Violence ... ``S22`` Immoral/Unethical). Prompted to emit
  ``Prompt harm: harmful/unharmful`` and ``Response Harm: harmful/unharmful``; with
  ``think=True`` it first reasons inside ``<think>...</think>`` (stripped before parsing).
- ``nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3`` (Llama-3.1-8B base) — NVIDIA's own
  published prompt over a 23-category taxonomy (``Other`` is inserted at ``S14``, shifting
  the rest down one), answering with a JSON object holding ``"User Safety"``,
  ``"Response Safety"``, and ``"Safety Categories"``. Multilingual (9 trained languages, ~20
  zero-shot). It has no reasoning mode, so ``think=True`` is rejected.

Verdict mapping onto ``GuardrailOutput`` (identical for both variants):

- ``valid`` is ``False`` when either the prompt or the response is judged harmful.
- ``categories`` carries two boolean signals — ``prompt_harm`` and ``response_harm``
  (``triggered`` reflects each verdict).
- ``explanation`` is the raw generation (including any ``<think>`` reasoning).
- ``usage`` carries the prompt / completion token counts. No canonical ``score`` or ``spans``
  are produced.
- Fails closed (``valid=False`` with ``extra={"parse_failure": True}``) when the prompt verdict
  is missing, or when a response was judged but its verdict did not parse.
- On the 8B-v3 variant only, ``extra["safety_categories"]`` additionally lists the violated
  taxonomy category names the model reported.

Expected inputs: a single ``input_text`` prompt string plus an optional ``output_text``
assistant response; when ``output_text`` is given the response is moderated alongside the
prompt. Single strings only — passing a list raises ``TypeError``.

The 4B variant is distributed under the NVIDIA Open Model License and the Gemma Terms of Use;
the 8B-v3 variant under the NVIDIA Open Model License and the Llama 3.1 Community License.

For more information, see the
[nvidia/Nemotron-Content-Safety-Reasoning-4B model card](https://huggingface.co/nvidia/Nemotron-Content-Safety-Reasoning-4B)
and the
[nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3 model card](https://huggingface.co/nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3).

## Supported Models

- `nvidia/Nemotron-Content-Safety-Reasoning-4B`
- `nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3`

## Constructor

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `think` | `bool` | No | `False` | If ``True``, request chain-of-thought reasoning (``/think``) before the verdict; otherwise ``/no_think``. Slower but can improve borderline judgments; the reasoning is stripped before parsing but kept in ``GuardrailOutput.explanation``. Supported only by the 4B variant. |
| `model_id` | `str | None` | No | `None` | Optional HuggingFace model ID; must be one of ``SUPPORTED_MODELS``. Defaults to ``nvidia/Nemotron-Content-Safety-Reasoning-4B``. |
| `provider` | `Provider[dict[str, Any], dict[str, Any]] | None` | No | `None` | Optional pre-configured provider. When ``None``, a ``HuggingFaceProvider`` is built targeting a causal LM (``AutoModelForCausalLM`` + ``AutoTokenizer``). A supplied ``HuggingFaceProvider`` is corrected to those classes at load time; any other provider is used as-is. |

Initialize the Nemotron Content Safety guardrail.

## validate

Classify ``input_text`` and, optionally, an assistant ``output_text``.

**Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `input_text` | `str` | Yes | — | The user prompt to moderate. Single string only. |
| `output_text` | `str | None` | No | `None` | Optional assistant response moderated alongside the prompt. When provided, a missing or unparsable response verdict causes the guardrail to fail closed. |

**Returns:** `GuardrailOutput`

## Benchmarks

### Content Safety

| Dataset (rev) | Metric | Threshold | Value | Harness | Source | Contam. |
| --- | --- | --- | --- | --- | --- | --- |
| openai_moderation (unspecified) | f1 | native-valid | 0.82392 | guardrail-bench+ag0.7.4 | measured:guardrail-bench+ag0.7.4 |  |
| xstest (unspecified) | fpr | native-valid | 0.26 | guardrail-bench+ag0.7.4 | measured:guardrail-bench+ag0.7.4 |  |
| wildguardmix (unspecified) | f1 | native-valid | 0.888136 | guardrail-bench+ag0.7.4 | measured:guardrail-bench+ag0.7.4 |  |
| aegis (unspecified) | f1 | native-valid | 0.853333 | guardrail-bench+ag0.7.4 | measured:guardrail-bench+ag0.7.4 | ⚠️ |
| jbb (unspecified) | f1 | native-valid | 0.834043 | guardrail-bench+ag0.7.4 | measured:guardrail-bench+ag0.7.4 |  |
| orbench (unspecified) | fpr | native-valid | 0.673684 | guardrail-bench+ag0.7.4 | measured:guardrail-bench+ag0.7.4 |  |

## License

- **Vendor:** NVIDIA
- **Default license:** `gemma` (of the default model/service)

| Model variant | License |
| --- | --- |
| `nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3` | `llama-3.1` |
| `nvidia/Nemotron-Content-Safety-Reasoning-4B` | `gemma` |
