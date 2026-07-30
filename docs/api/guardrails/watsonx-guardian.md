# WatsonxGuardian

Hosted text-detection moderation API running configurable Granite Guardian detectors.

This is the hosted, pay-per-use counterpart to the locally-run
:class:`~any_guardrail.guardrails.granite_guardian.granite_guardian.GraniteGuardian`
guardrail: the same Granite Guardian risk-detection family, served as a
purpose-built detection endpoint instead of running the weights yourself.

The ``Guardian`` class (from the ``ibm-watsonx-ai`` SDK) screens text against
a configurable set of detectors. The default ``granite_guardian`` detector
covers the Granite Guardian risk catalogue (harm, social bias, violence,
jailbreak, profanity, sexual content, plus RAG groundedness / relevance);
``hap`` (hate-abuse-profanity) and ``pii`` detectors are also available. Each
detector returns zero or more *detections*, each locating a risky span with a
score.

Auth is via an IBM Cloud IAM API key plus a region URL and a project (or
space). Obtain a key and project from https://dataplatform.cloud.ibm.com/ and
set them via ``WATSONX_APIKEY`` / ``WATSONX_URL`` / ``WATSONX_PROJECT_ID``
(or ``WATSONX_SPACE_ID``), or pass them directly. A free Lite plan is
available.

``GuardrailOutput`` mapping:
    - ``valid = no detections were returned`` (the detection API only returns
      detections at or above the configured threshold).
    - ``score`` is the highest detection score; ``0.0`` when nothing was
      detected.
    - ``categories`` lists one ``CategoryResult`` per detection (``name`` =
      the detected risk, ``triggered=True``, ``score`` = the detection score).
    - ``spans`` lists one ``SpanResult`` per detection that carries character
      offsets (watsonx detections locate the flagged substring).
    - ``raw`` is the full response dict from ``Guardian.detect``.

Expected input: ``validate`` takes a single string, ``content``, and screens it
against the configured detectors. There is no separate prompt-vs-response
argument; RAG groundedness / relevance are enabled by adding the corresponding
detector to ``detectors`` rather than by passing extra arguments here.

Research backing:
    - Padhi et al., *Granite Guardian* (https://arxiv.org/abs/2412.07724, 2024).
    - IBM tutorial: https://www.ibm.com/think/tutorials/llm-safeguards-granite-guardian-risk-detection
    - SDK reference: https://ibm.github.io/watsonx-ai-python-sdk/fm_text_detection.html

For more information, see:

- [watsonx.ai platform (API key / project, free Lite plan)](https://dataplatform.cloud.ibm.com/)
- [IBM tutorial: LLM safeguards with Granite Guardian risk detection](https://www.ibm.com/think/tutorials/llm-safeguards-granite-guardian-risk-detection)
- [watsonx.ai Python SDK: foundation-model text detection](https://ibm.github.io/watsonx-ai-python-sdk/fm_text_detection.html)
- [Granite Guardian (arXiv:2412.07724)](https://arxiv.org/abs/2412.07724)

## Supported Models

- `granite_guardian`

## Constructor

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `api_key` | `str | None` | No | `None` | IBM Cloud IAM API key. If ``None``, it is read from the ``WATSONX_APIKEY`` environment variable. Ignored when ``api_client`` is supplied. |
| `url` | `str | None` | No | `None` | watsonx.ai region endpoint, e.g. ``https://us-south.ml.cloud.ibm.com``. If ``None``, it is read from ``WATSONX_URL``. Ignored when ``api_client`` is supplied. |
| `project_id` | `str | None` | No | `None` | watsonx project ID. If ``None``, it is read from ``WATSONX_PROJECT_ID``. One of ``project_id`` / ``space_id`` is required (unless ``api_client`` is supplied). |
| `space_id` | `str | None` | No | `None` | watsonx deployment space ID. If ``None``, it is read from ``WATSONX_SPACE_ID``. Alternative to ``project_id``. |
| `detectors` | `dict[str, Any] | None` | No | `None` | Detector configuration forwarded to ``Guardian``. Defaults to ``{"granite_guardian": {}}``; pass e.g. ``{"granite_guardian": {"threshold": 0.6}, "pii": {}}`` to tune thresholds or add the ``hap`` / ``pii`` detectors. |
| `api_client` | `APIClient | None` | No | `None` | A pre-built ``ibm_watsonx_ai.APIClient``. When supplied, the credential arguments above are ignored and the client is used as-is (useful for shared clients or testing). |

Initialize the guardrail and build the watsonx ``Guardian`` client.

Building the client performs IAM authentication, so unlike the pure-REST
API guardrails this constructor does contact IBM Cloud (unless a
pre-built ``api_client`` is supplied).

## validate

Screen ``content`` against the configured watsonx detectors.

**Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `content` | `str` | Yes | — | The text to screen. |

**Returns:** `GuardrailOutput`
