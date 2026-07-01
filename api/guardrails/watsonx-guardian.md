# WatsonxGuardian

Wraps IBM watsonx.ai's Text Detection / ``Guardian`` moderation API.

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

Research backing:
    - Padhi et al., *Granite Guardian* (https://arxiv.org/abs/2412.07724, 2024).
    - IBM tutorial: https://www.ibm.com/think/tutorials/llm-safeguards-granite-guardian-risk-detection
    - SDK reference: https://ibm.github.io/watsonx-ai-python-sdk/fm_text_detection.html

Args:
    api_key (str | None): IBM Cloud IAM API key. Falls back to ``WATSONX_APIKEY``.
    url (str | None): watsonx.ai region endpoint (e.g.
        ``https://us-south.ml.cloud.ibm.com``). Falls back to ``WATSONX_URL``.
    project_id (str | None): watsonx project ID. Falls back to
        ``WATSONX_PROJECT_ID``. One of ``project_id`` / ``space_id`` is required.
    space_id (str | None): watsonx deployment space ID. Falls back to
        ``WATSONX_SPACE_ID``.
    detectors (dict | None): Detector configuration forwarded to ``Guardian``.
        Defaults to ``{"granite_guardian": {}}``. Pass e.g.
        ``{"granite_guardian": {"threshold": 0.6}, "pii": {}}`` to tune it.
    api_client (APIClient | None): A pre-built ``ibm_watsonx_ai.APIClient``.
        When supplied, the credential arguments above are ignored and the
        client is used as-is (useful for shared clients or testing).

## Supported Models

- `granite_guardian`

## Constructor

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `api_key` | `str | None` | No | `None` |
| `url` | `str | None` | No | `None` |
| `project_id` | `str | None` | No | `None` |
| `space_id` | `str | None` | No | `None` |
| `detectors` | `dict[str, Any] | None` | No | `None` |
| `api_client` | `APIClient | None` | No | `None` |

Initialize the guardrail and build the watsonx ``Guardian`` client.

Building the client performs IAM authentication, so unlike the pure-REST
API guardrails this constructor does contact IBM Cloud (unless a
pre-built ``api_client`` is supplied).

## validate

Screen ``content`` against the configured watsonx detectors.

**Parameters**

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `content` | `str` | Yes | — |

**Returns:** `GuardrailOutput`
