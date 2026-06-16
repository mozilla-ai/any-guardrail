# PrivacyFilter

OpenAI Privacy Filter — token-classification PII / secrets detector.

A token classifier that flags spans of personal data (names, addresses, emails,
phone numbers, account numbers, secrets, etc.). ``validate`` returns the detected
character ``spans`` (with entity ``label`` and confidence ``score``); ``valid`` is
``True`` when nothing is flagged. Detected entity types are summarized in
``categories`` and the riskiest span score is surfaced in ``score``.

For more information, see the
[openai/privacy-filter model card](https://huggingface.co/openai/privacy-filter).

Note: the model ships a custom (sparse-MoE) bidirectional architecture loaded via
``trust_remote_code``; span extraction relies on the tokenizer exposing offset
mappings. Validate on real weights before production use.

Args:
    model_id: Optional HuggingFace model ID. Defaults to ``openai/privacy-filter``.
    provider: Optional pre-configured provider. Defaults to a ``HuggingFaceProvider``
        loading an ``AutoModelForTokenClassification`` with ``trust_remote_code=True``.

## Supported Models

- `openai/privacy-filter`

## Constructor

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `model_id` | `str | None` | No | `None` |
| `provider` | `Provider[dict[str, Any], dict[str, Any]] | None` | No | `None` |

Initialize the Privacy Filter guardrail.

## validate

Detect PII spans in ``input_text``.

Returns a :class:`GuardrailOutput` whose ``spans`` lists every detected PII span
(character offsets, entity ``label``, and confidence ``score``). ``valid`` is
``True`` when no PII is found.

**Parameters**

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `input_text` | `str` | Yes | — |

**Returns:** `GuardrailOutput`
