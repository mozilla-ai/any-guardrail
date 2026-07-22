# GliNerPii

Span-level PII/NER detector emitting character spans and a redacted copy of the text.

Wraps the ``gliner2`` library to run a GLiNER2 encoder that extracts personally
identifiable information as character-offset spans, in a single pass, for a
caller-supplied (or default) list of entity types. Because GLiNER2 is zero-shot, the
``entity_types`` are specified at call time (defaulting to ``DEFAULT_PII_ENTITIES``),
so the same model can detect an arbitrary PII taxonomy without retraining.

Verdict mapping onto ``GuardrailOutput``:

- ``spans`` marks each detected entity (character ``start`` / ``end`` offsets into the
  input, the offending ``text``, ``label`` = the entity type, and a confidence ``score``).
- ``valid`` is ``True`` when no PII span is found.
- ``score`` is the maximum span confidence (higher = riskier; ``None`` when nothing is flagged).
- ``categories`` holds one entry per detected entity type (``triggered=True``).
- ``modified_text`` is a redacted copy of the input with each detected span replaced by
  ``redaction_placeholder`` (``None`` when nothing is flagged), so a downstream redact
  mitigation can consume it directly.

Expected inputs: a single text string. Extra keyword arguments to ``validate`` are ignored.

This guardrail wraps an upstream library rather than a provider, so it requires the
``gliner`` extra (``pip install 'any-guardrail[gliner]'``); the constructor re-raises a
helpful ``ImportError`` when it is missing.

For more information, see:

- [gliner2-privacy-filter-PII-multi model card](https://huggingface.co/fastino/gliner2-privacy-filter-PII-multi) (default).
- [GLiNER2-PII: A Multilingual Model for Personally Identifiable Information Extraction
  (arXiv:2605.09973)](https://arxiv.org/abs/2605.09973).


Raises:
    ImportError: When the ``gliner`` extra is not installed.

## Supported Models

- `fastino/gliner2-privacy-filter-PII-multi`

## Constructor

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model_id` | `str | None` | No | `None` | Optional HuggingFace model ID. Must be one of ``SUPPORTED_MODELS``; defaults to ``fastino/gliner2-privacy-filter-PII-multi``. |
| `threshold` | `float` | No | `0.5` | Default confidence threshold forwarded to ``gliner2``'s ``extract_entities``. Defaults to 0.5; a per-call ``threshold`` overrides it. |

Initialize the GLiNER2 PII guardrail.

## validate

Detect PII spans in ``input_text`` and emit a redacted copy.

**Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `input_text` | `str` | Yes | — | The text to scan for PII. A single string. |
| `entity_types` | `list[str] | None` | No | `None` | PII entity types to detect. Defaults to ``DEFAULT_PII_ENTITIES``. |
| `threshold` | `float | None` | No | `None` | Confidence threshold for this call. Defaults to the instance ``threshold``. |
| `redaction_placeholder` | `str` | No | `"[REDACTED_{label}]"` | Template used to replace each detected span in ``modified_text``; the substring ``{label}`` is replaced with the upper-cased entity type (e.g. ``"[REDACTED_EMAIL]"``). Any other braces are kept literal, so an arbitrary placeholder never raises. |

**Returns:** `GuardrailOutput`
