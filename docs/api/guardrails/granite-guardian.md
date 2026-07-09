# GraniteGuardian

Granite Guardian — hybrid-thinking safety and judge model covering harm, RAG groundedness, and function-calling risks via bring-your-own-criteria (IBM).

Granite Guardian is a decoder-LLM safeguard, derived from IBM's Granite models, that
evaluates whether a given text meets a single user-specified criterion. It runs through
``provider.generate_chat`` so it can be served from either a ``HuggingFaceProvider`` or a
``LlamafileProvider``. It supports:

- **Bring-Your-Own-Criteria (BYOC)**: arbitrary natural-language criteria.
- **Predefined risks**: see :class:`GraniteGuardianRisk` for ready-made strings covering
  safety (harm, social bias, jailbreak, violence, profanity, unethical behavior),
  RAG hallucination (groundedness, context relevance, answer relevance), and
  function-calling hallucination.
- **RAG evaluation**: pass ``documents`` to :meth:`validate` to check groundedness,
  context relevance, or answer relevance.
- **Function-calling evaluation**: pass ``available_tools`` to :meth:`validate` to
  check for function-calling hallucinations.
- **Think / no-think modes**: set ``think=True`` to request chain-of-thought reasoning
  before the verdict (higher latency, longer output); the default emits the verdict
  immediately.

The model answers ``yes`` when the text **meets** the criterion and ``no`` when it does
not. Criteria are phrased as *violations* (e.g. ``"the text contains harm"``), so the
verdict maps onto ``GuardrailOutput`` as:

- ``valid`` is ``True`` when the model answers ``no`` (violation absent = safe) and
  ``False`` when it answers ``yes`` (violation present). Phrase custom criteria
  accordingly.
- ``categories`` holds one ``CategoryResult`` named after the criterion, with
  ``triggered=True`` when the model answered ``yes``.
- ``score`` (the canonical numeric risk axis, where higher = riskier) is left ``None`` —
  the verdict is a binary yes/no rather than a graded probability.
- ``explanation`` is the full decoded generation, including any ``<think>...</think>``
  reasoning block in think mode; ``extra["raw_answer"]`` is the raw ``"yes"``/``"no"``
  string.
- When no verdict can be parsed the output fails closed: ``valid=False`` with
  ``extra={"parse_failure": True}``.

Expected inputs: a single ``input_text`` string (the user turn), plus an optional
``output_text`` (the assistant turn being judged), optional ``documents`` for RAG
criteria, and optional ``available_tools`` for function-call criteria. Only single-string
inputs are supported — a list input raises ``TypeError``.

For more information, see:

- [IBM Granite Guardian 4.1 8B model card](https://huggingface.co/ibm-granite/granite-guardian-4.1-8b)
- [Granite Guardian (arXiv:2412.07724)](https://arxiv.org/abs/2412.07724)
- [ibm-granite/granite-guardian on GitHub](https://github.com/ibm-granite/granite-guardian)


Raises:
    ValueError: If ``model_id`` is not in ``SUPPORTED_MODELS``.

## Supported Models

- `ibm-granite/granite-guardian-4.1-8b`

## Constructor

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `criteria` | `str` | Yes | — | The judging criterion applied to every ``validate`` call. Use a :class:`GraniteGuardianRisk` constant (e.g. ``GraniteGuardianRisk.HARM``, ``GraniteGuardianRisk.GROUNDEDNESS``) or a custom Bring-Your-Own-Criteria string. Phrase it as a violation (``"the text contains harmful content"``) so ``valid=True`` means the criterion was *not* met. |
| `think` | `bool` | No | `False` | If ``True``, run in think mode — the model emits a ``<think>...</think>`` chain-of-thought block before the ``<score>`` verdict (up to 2048 new tokens, higher latency). Defaults to ``False``, which returns the verdict immediately (up to 48 new tokens). |
| `model_id` | `str | None` | No | `None` | Optional HuggingFace model ID; must be one of ``SUPPORTED_MODELS``. Defaults to ``ibm-granite/granite-guardian-4.1-8b``. |
| `provider` | `Provider[dict[str, Any], dict[str, Any]] | None` | No | `None` | Optional pre-configured provider. When ``None``, a ``HuggingFaceProvider`` is built targeting ``AutoModelForCausalLM`` / ``AutoTokenizer`` (transformers is imported lazily here). Pass a ``LlamafileProvider`` to run a GGUF build without the huggingface extra; a supplied ``HuggingFaceProvider`` is re-pointed at the causal-LM classes for this load without mutating its constructor defaults. |

Initialize the Granite Guardian guardrail.

## validate

Score ``input_text`` (and optionally ``output_text``) against ``self.criteria``.

**Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `input_text` | `str` | Yes | — | The user turn. When ``output_text`` is also supplied, the assistant turn is the text being judged; otherwise the user turn is judged. |
| `output_text` | `str | None` | No | `None` | Optional assistant response. Required for criteria that judge the assistant (e.g. groundedness, answer relevance, function-call hallucination); omit to judge the user input directly (e.g. jailbreak, harm, context relevance). |
| `documents` | `list[dict[str, Any]] | None` | No | `None` | Optional RAG documents (dicts with ``doc_id`` and ``text``). Required for groundedness and context-relevance criteria. |
| `available_tools` | `list[dict[str, Any]] | None` | No | `None` | Optional tool definitions (dicts with ``name``, ``description``, ``parameters``). Required for function-call hallucination checks. |

**Returns:** `GuardrailOutput`
