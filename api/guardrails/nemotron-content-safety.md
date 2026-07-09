# NemotronContentSafety

Nemotron Content Safety — 4B reasoning safety classifier covering a 22-category content-safety taxonomy (NVIDIA).

Decoder LLM (Gemma-3-4B base) that classifies a user prompt and an optional assistant response
against NVIDIA's 22-category content-safety taxonomy (``S1`` Violence ... ``S22``
Immoral/Unethical). The model is prompted to emit ``Prompt harm: harmful/unharmful`` and
``Response Harm: harmful/unharmful``; with ``think=True`` it first reasons inside
``<think>...</think>`` (stripped before the verdict is parsed).

Verdict mapping onto ``GuardrailOutput``:

- ``valid`` is ``False`` when either the prompt or the response is judged harmful.
- ``categories`` carries two boolean signals — ``prompt_harm`` and ``response_harm``
  (``triggered`` reflects each verdict).
- ``explanation`` is the raw generation (including any ``<think>`` reasoning).
- ``usage`` carries the prompt / completion token counts. No canonical ``score`` or ``spans``
  are produced.
- Fails closed (``valid=False`` with ``extra={"parse_failure": True}``) when the prompt verdict
  is missing, or when a response was judged but its verdict did not parse.

Expected inputs: a single ``input_text`` prompt string plus an optional ``output_text``
assistant response; when ``output_text`` is given the response is moderated alongside the
prompt. Single strings only — passing a list raises ``TypeError``.

Distributed under the NVIDIA Open Model License and the Gemma Terms of Use.

For more information, see the
[nvidia/Nemotron-Content-Safety-Reasoning-4B model card](https://huggingface.co/nvidia/Nemotron-Content-Safety-Reasoning-4B).

## Supported Models

- `nvidia/Nemotron-Content-Safety-Reasoning-4B`

## Constructor

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `think` | `bool` | No | `False` | If ``True``, request chain-of-thought reasoning (``/think``) before the verdict; otherwise ``/no_think``. Slower but can improve borderline judgments; the reasoning is stripped before parsing but kept in ``GuardrailOutput.explanation``. |
| `model_id` | `str | None` | No | `None` | Optional HuggingFace model ID; must be one of ``SUPPORTED_MODELS``. Defaults to ``nvidia/Nemotron-Content-Safety-Reasoning-4B``. |
| `provider` | `Optional[Provider[dict[str, Any], dict[str, Any]]]` | No | `None` | Optional pre-configured provider. When ``None``, a ``HuggingFaceProvider`` is built targeting a causal LM (``AutoModelForCausalLM`` + ``AutoTokenizer``). A supplied ``HuggingFaceProvider`` is corrected to those classes at load time; any other provider is used as-is. |

Initialize the Nemotron Content Safety guardrail.

## validate

Classify ``input_text`` and, optionally, an assistant ``output_text``.

**Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `input_text` | `str` | Yes | — | The user prompt to moderate. Single string only. |
| `output_text` | `str | None` | No | `None` | Optional assistant response moderated alongside the prompt. When provided, a missing or unparsable response verdict causes the guardrail to fail closed. |

**Returns:** `GuardrailOutput`
