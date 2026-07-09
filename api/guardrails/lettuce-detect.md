# LettuceDetect

LettuceDetect — token/span-level RAG hallucination detector built on ModernBERT (KRLabs).

Wraps the ``lettucedetect`` library to flag spans of an answer that are not supported
by the provided context. Token classification over the (context, question, answer)
triple lets it exploit ModernBERT's long context window, so full RAG contexts fit in
a single pass. The supported checkpoints are English (``-en-v1``).

Expected inputs: ``validate(input_text, context=..., question=...)`` treats
``input_text`` as the *answer* to check; ``context`` (a string or list of strings) is
the grounding text and is semantically required — omitting it raises ``ValueError``;
``question`` is optional. Single answer per call (no batching).

``GuardrailOutput`` mapping: ``spans`` mark hallucinated text (character ``start`` /
``end`` offsets into the answer, the offending ``text``, ``label="hallucination"``,
and a confidence ``score``); ``valid`` is ``True`` when no hallucinated span is
found; the top-level ``score`` is the maximum span confidence (higher = riskier,
``None`` when nothing was flagged); ``categories`` holds a single ``hallucination``
entry with ``triggered=True`` when any span was found. ``usage`` records the model ID
and latency.

For more information, see:

- [lettucedect-base-modernbert-en-v1](https://huggingface.co/KRLabsOrg/lettucedect-base-modernbert-en-v1) (default).
- [lettucedect-large-modernbert-en-v1](https://huggingface.co/KRLabsOrg/lettucedect-large-modernbert-en-v1).
- [LettuceDetect on GitHub](https://github.com/KRLabsOrg/LettuceDetect).
- [LettuceDetect: A Hallucination Detection Framework for RAG Applications
  (arXiv:2502.17125)](https://arxiv.org/abs/2502.17125).


Raises:
    ImportError: When the ``lettucedetect`` extra is not installed.

## Supported Models

- `KRLabsOrg/lettucedect-base-modernbert-en-v1`
- `KRLabsOrg/lettucedect-large-modernbert-en-v1`

## Constructor

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model_id` | `str | None` | No | `None` | Optional HuggingFace model ID; one of ``SUPPORTED_MODELS`` (``KRLabsOrg/lettucedect-base-modernbert-en-v1`` or ``KRLabsOrg/lettucedect-large-modernbert-en-v1``). Defaults to the base model; the large model trades speed for accuracy. |
| `method` | `str` | No | `"transformer"` | Detection method forwarded to ``lettucedetect``'s ``HallucinationDetector`` constructor. Defaults to ``"transformer"``, the encoder token-classification method the supported checkpoints were trained for. |

Initialize the LettuceDetect guardrail.

## validate

Detect hallucinated spans in ``input_text`` (the answer) against ``context``.

**Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `input_text` | `str` | Yes | — | The answer to check for hallucinations. |
| `context` | `str | list[str] | None` | No | `None` | The grounding context (a string or list of strings). Required. |
| `question` | `str | None` | No | `None` | Optional question the answer responds to. |

**Returns:** `GuardrailOutput`
