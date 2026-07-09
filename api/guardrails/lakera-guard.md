# LakeraGuard

Lakera Guard — hosted API for prompt-injection, jailbreak, content-moderation, and PII detection (Lakera).

Lakera Guard exposes a single ``/v2/guard`` endpoint that returns whether a message (or message list)
was flagged. ``validate(content)`` accepts either a plain string (wrapped as a single user-role
message) or a pre-formed chat-message list (``[{"role": "user", "content": "..."}]``), so both
prompts and full conversations (including assistant turns) can be screened. By default this
guardrail also opts into the endpoint's richer outputs so callers get the full picture of *why*
something was flagged:

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
      level to a float (``l1_confident`` → ``1.0`` … ``l5_unlikely`` → ``0.2``, higher = riskier);
      ``0.0`` when nothing was detected. If ``breakdown`` is disabled, ``score`` falls back to
      ``1.0`` when flagged else ``0.0``.
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

For more information, see:

- [Lakera platform (API keys, free Community tier)](https://platform.lakera.ai/)
- [Product overview: prompt defense](https://www.lakera.ai/prompt-defense)
- [API docs: Guard](https://docs.lakera.ai/docs/api/guard)
- [API reference: screen content (`/v2/guard`)](https://docs.lakera.ai/api-reference/lakera-api/guard/screen-content)
- [Gandalf the Red: Adaptive Security for LLMs (arXiv:2501.07927)](https://arxiv.org/abs/2501.07927)

Brand transition note:
    Lakera was acquired by Cisco in 2025 and is being folded into Cisco AI Defense. The
    ``docs.lakera.ai`` API remains the public surface for now, with Pro/Enterprise tiers
    sales-gated through Cisco. Endpoint consolidation under Cisco AI Defense is expected within
    12-18 months; expect the constructor's ``endpoint`` default to be revised at that point.

## Supported Models

- `lakera-guard`

## Constructor

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `api_key` | `str | None` | No | `None` | API key for authenticating with the Lakera Guard API. If not provided, it is read from the ``LAKERA_API_KEY`` environment variable. Obtain one at https://platform.lakera.ai/ (free Community tier: 10k requests/month). |
| `endpoint` | `str` | No | `"https://api.lakera.ai/v2/guard"` | Lakera Guard API endpoint URL. Defaults to the v2 endpoint at ``https://api.lakera.ai/v2/guard``; override for self-hosted or regional deployments. |
| `project_id` | `str | None` | No | `None` | Optional Lakera project ID (e.g. ``"project-1234"``). Projects carry per-project policy configuration (which detectors run, severity thresholds, custom rules); when supplied, it is forwarded with each request so that project's policy is applied. |
| `breakdown` | `bool` | No | `True` | If ``True`` (default), request the per-detector ``breakdown`` list, which also enables the graded ``score`` / ``categories`` mapping. If ``False``, ``score`` degrades to ``1.0``/``0.0`` and ``categories`` is empty. |
| `payload` | `bool` | No | `True` | If ``True`` (default), request the ``payload`` list locating PII / profanity / custom-regex matches (``start`` / ``end`` offsets, matched ``text``, ``labels``), surfaced in ``extra["payload"]``. |
| `dev_info` | `bool` | No | `False` | If ``True``, request Lakera build information (git revision, model version) in the response, surfaced in ``extra["dev_info"]``. Defaults to ``False``. |
| `metadata` | `dict[str, Any] | None` | No | `None` | Optional request metadata forwarded to Lakera for observability, e.g. ``{"user_id": "u-42", "session_id": "s-1"}``. |

Initialize the Lakera Guard guardrail with the provided configuration.

Does not perform any network I/O — the API is only contacted when ``validate()`` is called.

## validate

Validate a string or chat-message list against the Lakera Guard API.

**Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `content` | `str | list[dict[str, str]]` | Yes | — | Either a plain string (wrapped as a single user-role message, the common prompt-screening case) or a pre-formed list of chat messages following the ``[{"role": "user", "content": "..."}]`` shape. Pass the message-list form to screen a whole conversation, e.g. ``[{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "..."}]``. |

**Returns:** `GuardrailOutput`
