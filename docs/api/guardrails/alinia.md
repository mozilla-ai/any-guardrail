# Alinia

Alinia — hosted content-moderation and safety-detection API with configurable detection policies (Alinia AI).

Sends a text input or a full conversation to the Alinia API, which runs whichever detections
you enable via ``detection_config`` (e.g. ``{"security": True}`` for prompt-injection /
data-exfiltration detection, plus safety, compliance, and hallucination policies) and reports
per-category verdicts. Alinia's detection models are multilingual (its security model is
trained on English, Spanish, and Catalan).

Verdict mapping: ``valid`` is ``True`` when Alinia does not flag the input. ``categories``
flattens Alinia's nested ``category_details`` into one entry per ``group/label`` — boolean
details become ``triggered`` flags, numeric details become per-category ``score`` values.
The top-level ``score`` is the highest numeric category score (higher = riskier), or ``None``
when the endpoint returns only booleans. ``explanation`` carries the recommendation text when
the endpoint returns one (e.g. sensitive information), ``action`` carries a structured
recommendation's action (e.g. ``"block"``), and ``raw`` is the full response JSON.

Expected inputs: ``validate`` accepts either a plain string or a list of chat-message dicts
(``{"role": ..., "content": ...}``), plus an optional model ``output`` and optional
``context_documents`` for detections that evaluate responses in context.

You must obtain an API key and the endpoint URL from Alinia, and pass them either directly to
the constructor or via the ``ALINIA_API_KEY`` / ``ALINIA_ENDPOINT`` environment variables.

For more information, see:

- [Alinia AI](https://alinia.ai/) (vendor site).
- [Integrating Alinia into any-guardrail](https://blog.mozilla.ai/integrating-alinia-into-any-guardrail-for-multilingual-ai-security/)
  (Mozilla AI blog walkthrough of this guardrail).

## Constructor

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `detection_config` | `str | dict[str, float | bool] | dict[str, dict[str, float | bool | str]]` | Yes | — | Which detections to run and their thresholds. Pass either a detection-configuration ID string registered with Alinia, or a dict enabling detections inline — e.g. ``{"security": True}``, or nested per-policy settings such as ``{"safety": {"toxicity": 0.8}}``. |
| `api_key` | `str | None` | No | `None` | Alinia API key. If ``None``, it is read from the ``ALINIA_API_KEY`` environment variable. |
| `endpoint` | `str | None` | No | `None` | Alinia API endpoint URL (obtained from Alinia alongside the key). If ``None``, it is read from the ``ALINIA_ENDPOINT`` environment variable. |
| `metadata` | `dict[str, Any] | None` | No | `None` | Optional metadata dict sent with every request (e.g. app or user identifiers for Alinia-side monitoring). |
| `blocked_response` | `dict[str, str] | None` | No | `None` | Optional response Alinia should return when content is blocked, e.g. ``{"output": "Sorry, I can't help with that."}``. |
| `stream` | `bool` | No | `False` | Whether to request a streaming API response. Defaults to ``False``. |

Initialize the Alinia guardrail with the provided configuration.

## validate

Validate conversation or text input using the Alinia API.

This can be used for validation using any of the API endpoints provided by Alinia.

**Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `conversation` | `str | list[dict[str, str]]` | Yes | — | The input to validate. Either a plain string (sent as ``input``, e.g. ``"Ignore all instructions and ..."``) or a list of chat-message dicts (sent as ``messages``), e.g. ``[{"role": "user", "content": "..."}]``. |
| `output` | `str | None` | No | `None` | Optional model response to validate alongside the input, for detections that evaluate outputs (e.g. hallucination or compliance checks). |
| `context_documents` | `list[str] | None` | No | `None` | Optional context documents (e.g. retrieved RAG passages) that give output-side detections the grounding text to check against. |

**Returns:** `GuardrailOutput`
