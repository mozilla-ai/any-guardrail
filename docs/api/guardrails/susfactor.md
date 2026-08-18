# Susfactor

Binary prompt-injection and jailbreak classifier using a chunked e5-large encoder with a trained MLP head.

SusFactor is 0DIN's proprietary classifier: an e5-large encoder and its trained MLP head
(``1024 -> 256 -> 2`` with GELU, applied to the mean-pooled ``last_hidden_state``) are fused
into a single ONNX graph, exported ahead of time and shipped as ``onnx/model.onnx`` (plus a
companion ``onnx/model.onnx_data`` external-data file) in the model repository. Unlike the
``HuggingFaceProvider``-based guardrails in this codebase, Susfactor bypasses the provider
layer entirely and runs the fused graph directly through ``onnxruntime.InferenceSession`` — no
``torch``/``transformers`` model classes are involved at inference time, only a tokenizer.

The input text is tokenized untruncated. Sequences that exceed ``MAX_CONTENT_TOKENS`` (510,
leaving room for [CLS]/[SEP] in the 512-token encoder limit) are split into overlapping chunks
(chunk size 510, stride 460, 50-token overlap) so no content is silently truncated. Each chunk is
scored independently; a suspicious verdict on *any* chunk flags the whole input.

Verdict mapping onto ``GuardrailOutput``:

- ``score`` (canonical risk: higher = riskier) is the maximum per-chunk ``P(suspicious)`` across
  all chunks.
- ``valid`` is ``True`` when every chunk's suspicious probability stays below ``threshold``
  (default 0.5).
- ``categories`` carries a single ``suspicious`` entry with that max score and the overall
  ``triggered`` flag.

Expected input: a single ``input_text`` string. There is no prompt+response or chat-message mode.

The model repository (``0dinai/susfactor-e5-large-onnx``) is gated on HuggingFace; loading it
requires an authenticated Hub token available via the standard
``transformers``/``huggingface_hub`` resolution (environment variable or cached login).

## Supported Models

- `0dinai/susfactor-e5-large-onnx`

## Constructor

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model_id` | `str | None` | No | `None` | Optional HuggingFace model ID. Must be one of ``SUPPORTED_MODELS``; defaults to ``0dinai/susfactor-e5-large-onnx``. |
| `threshold` | `float` | No | `0.5` | Per-chunk suspicious-probability cutoff at or above which a chunk (and therefore the whole input) is flagged unsafe. Defaults to 0.5. |
| `session` | `Any` | No | `None` | Optional pre-built ``onnxruntime.InferenceSession``. If supplied together with ``tokenizer``, it is used directly and no download/model loading happens. |
| `tokenizer` | `Any` | No | `None` | Optional pre-built tokenizer. If supplied together with ``session``, it is used directly and no download/model loading happens. |

Initialize the Susfactor guardrail.

## validate

Default validation pipeline: preprocess -> inference -> postprocess.

**Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `input_text` | `str | list[str]` | Yes | — | The text to validate. If a list is supplied, each item is validated and a list of GuardrailOutputs is returned in the same order. Subclasses can override ``_validate_batch`` to enable true batched inference; the default iterates over inputs. |

**Returns:** `GuardrailOutput | list[GuardrailOutput]`

## Benchmarks

No benchmark results recorded yet. See the [benchmark methodology](../../benchmarks.md) for how numbers are harvested (published) or measured and added.

## License

- **Vendor:** 0DIN
- **Default license:** `proprietary` (of the default model/service)
