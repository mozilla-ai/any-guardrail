# Susfactor

Binary prompt-injection and jailbreak classifier using a chunked e5-large encoder with a trained MLP head.

SusFactor is 0DIN's proprietary classifier: an e5-large encoder and its trained MLP head
(``1024 -> 256 -> 2`` with GELU, applied to the mean-pooled ``last_hidden_state``) are fused
into a single ONNX graph, exported ahead of time and shipped as ``onnx/model.onnx`` (plus a
companion ``onnx/model.onnx_data`` external-data file) in the model repository. By default
Susfactor bypasses the provider layer and runs the fused graph directly through
``onnxruntime.InferenceSession`` — no ``torch``/``transformers`` model classes are involved at
inference time, only a tokenizer.

The input text is tokenized untruncated. Sequences that exceed ``MAX_CONTENT_TOKENS`` (510,
leaving room for [CLS]/[SEP] in the 512-token encoder limit) are split into overlapping chunks
(chunk size 510, stride 460, 50-token overlap) so no content is silently truncated. Each chunk is
scored independently; a suspicious verdict on *any* chunk flags the whole input.

**Hosted backend.** The model repository is gated on HuggingFace, so 0DIN also serves the same
classifier over REST. Pass ``provider=ZeroDinProvider()`` (from
``any_guardrail.providers.zero_din``, credentials via ``api_key=`` or the ``ODIN_API_KEY``
environment variable) to run against 0DIN Defense instead — same class, same ``threshold``, same
``GuardrailOutput``, and no ``onnx`` extra required. Two differences are worth knowing: the
hosted API chunks server-side with unpublished size/stride parameters and returns only the
maximum chunk score, so the two backends can disagree on very long inputs; and its own
``is_suspicious`` flag is hardcoded at 0.5 and is ignored here — ``threshold`` still decides
``valid``, with the untouched response JSON available in ``raw``.

Any provider supplied here must return ``{"chunk_scores": list[float]}`` from ``infer()``.

Verdict mapping onto ``GuardrailOutput``:

- ``score`` (canonical risk: higher = riskier) is the maximum per-chunk ``P(suspicious)`` across
  all chunks.
- ``valid`` is ``True`` when every chunk's suspicious probability stays below ``threshold``
  (default 0.5).
- ``categories`` carries a single ``suspicious`` entry with that max score and the overall
  ``triggered`` flag.
- A backend that returns no scores at all fails closed: ``valid=False`` with
  ``extra={"parse_failure": True}``.

Expected input: a single ``input_text`` string. There is no prompt+response or chat-message
mode. List input is accepted via the inherited ``validate()``, which scores each item in turn —
on the hosted backend that is one HTTP request per item, since the API has no batch endpoint.

See https://0din.ai/docs/defense/introduction for SusFactor's model card, threat-feed
training methodology, and benchmark results, and https://0din.ai/docs/defense/api for the
hosted API.

The model repository (``0dinai/susfactor-e5-large-onnx``) is gated on HuggingFace; loading it
requires an authenticated Hub token available via the standard
``transformers``/``huggingface_hub`` resolution (environment variable or cached login).

## Supported Models

- `0dinai/susfactor-e5-large-onnx`
- `susfactor-api`

## Constructor

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model_id` | `str | None` | No | `None` | Optional model ID. Must be one of ``SUPPORTED_MODELS``; defaults to ``0dinai/susfactor-e5-large-onnx`` locally, or to the ``provider``'s own ``default_model_id`` when one is supplied. |
| `threshold` | `float` | No | `0.5` | Per-chunk suspicious-probability cutoff at or above which a chunk (and therefore the whole input) is flagged unsafe. Defaults to 0.5. |
| `provider` | `Provider[dict[str, Any], dict[str, Any]] | None` | No | `None` | Optional execution provider whose ``infer()`` returns ``{"chunk_scores": [...]}``. ``None`` runs the local ONNX graph; pass ``ZeroDinProvider()`` for 0DIN's hosted API. |
| `session` | `Any` | No | `None` | Optional pre-built ``onnxruntime.InferenceSession``. If supplied together with ``tokenizer``, it is used directly and no download/model loading happens. Ignored when ``provider`` is set. |
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
