# LakeraGuard

Wraps the Lakera Guard REST API for prompt-injection, jailbreak, content-moderation, and PII detection.

Lakera Guard exposes a single ``/v2/guard`` endpoint that returns whether a message (or message list)
was flagged. By default this guardrail also opts into the endpoint's richer outputs so callers get the
full picture of *why* something was flagged:

- ``breakdown`` (requested via ``breakdown=True``): one entry per detector the policy ran, with its
  ``detector_type``, whether it ``detected`` a threat, and an ordinal confidence ``result``
  (``l1_confident`` … ``l5_unlikely`` / ``no_level``).
- ``payload`` (requested via ``payload=True``): the string location (``start`` / ``end``), matched
  ``text``, ``detector_type``, and ``labels`` of any PII, profanity, or custom-regex matches.

Auth is via a bearer token; obtain an API key from https://platform.lakera.ai/ (free Community tier:
10k requests/month) and set it via the ``LAKERA_API_KEY`` environment variable or pass it directly.

``GuardrailOutput`` mapping:
    - ``valid = not flagged``.
    - ``score`` is the highest detector confidence among *detected* threats, mapped from the ordinal
      level to a float (``l1_confident`` → ``1.0`` … ``l5_unlikely`` → ``0.2``); ``0.0`` when nothing
      was detected. If ``breakdown`` is disabled, ``score`` falls back to ``1.0`` when flagged else
      ``0.0``.
    - ``categories`` lists one ``CategoryResult`` per ``breakdown`` entry (``name`` =
      ``detector_type``, ``triggered`` = whether it detected, ``score`` = the mapped confidence).
    - ``extra`` carries the ``flagged`` flag, the ``payload`` list, the request ``metadata``
      (``request_uuid``), the convenience ``detected_detector_types`` list, and ``dev_info``
      when requested.
    - ``raw`` is the full Lakera response body (including the per-detector ``breakdown``).

Research backing:
    - Pfister et al., *Gandalf the Red: Adaptive Security for LLMs*
      (https://arxiv.org/abs/2501.07927, 2025) introduces the D-SEC threat model and releases the
      279k-prompt Gandalf attack dataset that informs Lakera's training pipeline.
    - The differentiator vs. OSS DeBERTa-based prompt-injection detectors is the proprietary
      Gandalf-derived training data: 1M+ players and 80M+ adversarial prompts collected via
      Lakera's public Gandalf challenge platform. The Gandalf paper shows OSS detectors
      underperform on adaptive attacks at scale.
    - Product overview: https://www.lakera.ai/prompt-defense
    - API docs: https://docs.lakera.ai/docs/api/guard

Brand transition note:
    Lakera was acquired by Cisco in 2025 and is being folded into Cisco AI Defense. The
    ``docs.lakera.ai`` API remains the public surface for now, with Pro/Enterprise tiers
    sales-gated through Cisco. Endpoint consolidation under Cisco AI Defense is expected within
    12-18 months; expect the constructor's ``endpoint`` default to be revised at that point.

Args:
    api_key (str | None): The API key for authenticating with the Lakera Guard API. If not
        provided, it will be read from the ``LAKERA_API_KEY`` environment variable.
    endpoint (str): The Lakera Guard API endpoint URL. Defaults to the v2 endpoint at
        ``https://api.lakera.ai/v2/guard``.
    project_id (str | None): Optional Lakera project ID. Lakera projects allow per-project
        policy configuration (which detectors to run, severity thresholds, custom rules); when
        supplied, the project ID is forwarded with each request so the project's policy is applied.
    breakdown (bool): Request the per-detector ``breakdown`` list. Defaults to ``True``.
    payload (bool): Request the ``payload`` list locating PII / profanity / custom-regex matches.
        Defaults to ``True``.
    dev_info (bool): Request Lakera build information (git revision, model version) in the response.
        Defaults to ``False``.
    metadata (dict[str, Any] | None): Optional request metadata forwarded to Lakera for
        observability (e.g. ``user_id``, ``session_id``, ``ip_address``, ``internal_request_id``).

## Supported Models

- `lakera-guard`

## Constructor

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `api_key` | `str | None` | No | `None` |
| `endpoint` | `str` | No | `"https://api.lakera.ai/v2/guard"` |
| `project_id` | `str | None` | No | `None` |
| `breakdown` | `bool` | No | `True` |
| `payload` | `bool` | No | `True` |
| `dev_info` | `bool` | No | `False` |
| `metadata` | `dict[str, Any] | None` | No | `None` |

Initialize the Lakera Guard guardrail with the provided configuration.

Does not perform any network I/O — the API is only contacted when ``validate()`` is called.

## validate

Validate a string or chat-message list against the Lakera Guard API.

**Parameters**

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `content` | `str | list[dict[str, str]]` | Yes | — |

**Returns:** `GuardrailOutput`
