# LlamaGuard

Decoder-LLM safety classifier judging prompts and responses against the 14-category MLCommons hazard taxonomy.

Llama Guard is Meta's instruction-tuned safety model. Each call wraps a user prompt (and,
optionally, an assistant response) in the model's moderation template listing the 14 MLCommons
hazard categories (``S1`` Violent Crimes ... ``S14`` Code Interpreter Abuse), then generates a
verdict: ``safe``, or ``unsafe`` followed by the violated category codes. This wrapper supports
both Llama Guard 3 (1B / 8B, text-only) and Llama Guard 4 (12B, natively multimodal — this
integration passes text content only); the per-variant chat-template quirks (v3 evaluates the
conversation as-is without an appended assistant prefix, v4 uses the standard template) are
handled internally, so callers select a variant purely via ``model_id``. Llama Guard 3 is
trained for multilingual moderation across eight languages (English, French, German, Hindi,
Italian, Portuguese, Spanish, Thai).

Verdict mapping onto ``GuardrailOutput``:

- ``valid`` is ``False`` when the generation contains ``unsafe`` (case-insensitive), ``True``
  otherwise.
- ``categories`` lists the violated hazard codes parsed from the generation, one
  ``CategoryResult`` per code (``name`` = ``Sx``, ``description`` = the taxonomy label,
  ``triggered=True``), deduplicated in order of first appearance; empty when the verdict is
  ``safe``. Unknown codes are kept with ``description=None`` so taxonomy additions still surface.
- ``explanation`` is the raw generated text.
- ``usage`` carries the prompt / completion token counts.
- No canonical ``score`` and no ``spans`` are produced (``score`` is ``None``).

Expected inputs: a single ``input_text`` string (the user prompt) plus an optional
``output_text`` string (an assistant response). When ``output_text`` is supplied, the model
judges the full ``[user, assistant]`` turn — i.e. it moderates the response in the context of
the prompt. Single strings only; list / batch input is not supported.

The models are gated on HuggingFace and distributed under Meta's Llama Community License.

For more information, see:

- [Llama Guard 3 model card (Meta)](https://www.llama.com/docs/model-cards-and-prompt-formats/llama-guard-3/)
- [Llama Guard 4 model card (Meta)](https://www.llama.com/docs/model-cards-and-prompt-formats/llama-guard-4/)
- [meta-llama/Llama-Guard-3-1B](https://huggingface.co/meta-llama/Llama-Guard-3-1B)
- [meta-llama/Llama-Guard-3-8B](https://huggingface.co/meta-llama/Llama-Guard-3-8B)
- [meta-llama/Llama-Guard-4-12B](https://huggingface.co/meta-llama/Llama-Guard-4-12B)

## Supported Models

- `meta-llama/Llama-Guard-3-1B`
- `meta-llama/Llama-Guard-3-8B`
- `meta-llama/Llama-Guard-4-12B`

## Constructor

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model_id` | `str | None` | No | `None` | Optional HuggingFace model ID; must be one of ``SUPPORTED_MODELS``. Selects the variant — ``meta-llama/Llama-Guard-3-1B`` (default) or ``meta-llama/Llama-Guard-3-8B`` for Llama Guard 3, ``meta-llama/Llama-Guard-4-12B`` for Llama Guard 4. |
| `provider` | `Optional[Provider[dict[str, Any], dict[str, Any]]]` | No | `None` | Optional pre-configured provider. When ``None``, a ``HuggingFaceProvider`` is built with the correct model / tokenizer classes for the selected variant (a causal LM for v3; ``Llama4ForConditionalGeneration`` + ``AutoProcessor`` for v4). A supplied ``HuggingFaceProvider`` is corrected to the same classes at load time (without mutating it); any other provider (e.g. ``LlamafileProvider``) is used as-is. |

Initialize the Llama Guard guardrail.

## validate

Default validation pipeline: preprocess -> inference -> postprocess.

**Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `input_text` | `str | list[str]` | Yes | — | The text to validate. If a list is supplied, each item is validated and a list of GuardrailOutputs is returned in the same order. Subclasses can override ``_validate_batch`` to enable true batched inference; the default iterates over inputs. |

**Returns:** `GuardrailOutput | list[GuardrailOutput]`
