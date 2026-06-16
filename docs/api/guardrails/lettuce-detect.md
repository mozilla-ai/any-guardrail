# LettuceDetect

LettuceDetect — token/span-level RAG hallucination detector (KRLabs).

Wraps the ``lettucedetect`` library to flag spans of an answer that are not
supported by the provided context. ``validate(input_text, context=..., question=...)``
treats ``input_text`` as the answer; the returned ``spans`` mark hallucinated text
(character offsets + confidence ``score``). ``valid`` is ``True`` when no
hallucinated span is found.

For more information, see the model cards and library:

- [lettucedect-base-modernbert-en-v1](https://huggingface.co/KRLabsOrg/lettucedect-base-modernbert-en-v1) (default).
- [lettucedect-large-modernbert-en-v1](https://huggingface.co/KRLabsOrg/lettucedect-large-modernbert-en-v1).
- [LettuceDetect on GitHub](https://github.com/KRLabsOrg/LettuceDetect).

Args:
    model_id: Optional HuggingFace model ID. Defaults to the base ModernBERT model.
    method: Detection method passed to the library. Defaults to ``"transformer"``.

Raises:
    ImportError: When the ``lettucedetect`` extra is not installed.

## Supported Models

- `KRLabsOrg/lettucedect-base-modernbert-en-v1`
- `KRLabsOrg/lettucedect-large-modernbert-en-v1`

## Constructor

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `model_id` | `str | None` | No | `None` |
| `method` | `str` | No | `"transformer"` |

Initialize the LettuceDetect guardrail.

## validate

Detect hallucinated spans in ``input_text`` (the answer) against ``context``.

**Parameters**

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `input_text` | `str` | Yes | — |
| `context` | `str | list[str] | None` | No | `None` |
| `question` | `str | None` | No | `None` |

**Returns:** `GuardrailOutput`
