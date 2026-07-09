# AzurePromptShields

Azure AI Prompt Shields — hosted detector for direct (user prompt) and indirect (document-borne) prompt-injection and jailbreak attacks (Microsoft).

Prompt Shields is a service from Azure AI Content Safety that detects prompt-injection
and jailbreak attacks against LLM applications. It supports two attack surfaces:

- **Direct attacks (user_prompt)**: malicious instructions in end-user input attempting
  to override the system prompt, exfiltrate sensitive info, or otherwise jailbreak
  the model.
- **Indirect attacks (documents)**: data-borne prompt injection embedded inside
  retrieved documents, tool outputs, or other context fed to the model. Microsoft
  Research's [Spotlighting](https://arxiv.org/abs/2403.14720) technique (Hines et al., 2024)
  is the published basis for this indirect-attack detection.

Expected inputs: ``validate`` takes an optional ``user_prompt`` (a single string, the
end-user prompt) and/or optional ``documents`` (a list of strings — retrieved context,
tool outputs, etc.). At least one of the two must be provided; each surface is only
analyzed when its argument is supplied.

``GuardrailOutput`` mapping:
    - ``valid`` is ``True`` iff Azure detects no attack anywhere — neither in the user
      prompt nor in **any** supplied document.
    - ``score`` is a binary severity proxy: ``1.0`` when any attack is detected, ``0.0``
      otherwise (higher = riskier). Prompt Shields returns per-surface booleans rather
      than a continuous risk probability.
    - ``categories`` holds one ``CategoryResult`` per analyzed source — ``user_prompt``
      (when supplied) and ``document_{i}`` for each document — with ``triggered`` set to
      that surface's ``attackDetected`` flag.
    - ``extra`` carries the per-field detection booleans; ``raw`` is the full REST
      response. A malformed / unparsable Azure payload fails closed (``valid=False``,
      ``score=1.0``, ``extra={"parse_failure": True}``).

This guardrail hits the same Azure Content Safety resource as ``AzureContentSafety`` and
reuses the same env vars (``CONTENT_SAFETY_KEY`` / ``CONTENT_SAFETY_ENDPOINT``) and the
``azure-content-safety`` optional extra. It is, however, a separate guardrail because the
threat surface, request payload, and response shape differ from the content-harm endpoint.

Implementation note:
    The current ``azure-ai-contentsafety`` Python SDK (``>=1.0.0``) does not yet expose a
    ``shield_prompt`` method on ``ContentSafetyClient``. This guardrail therefore calls the
    Prompt Shields REST endpoint directly via ``requests.post`` using the API version
    ``2024-09-01``. If a future SDK release adds first-class support, this class can be
    switched over without changing the public ``validate()`` signature.

Caveat on real-world robustness:
    An independent evaluation
    ([arXiv:2504.11168](https://arxiv.org/pdf/2504.11168), 2025) reports large evasion gaps
    for hosted prompt-injection detectors, including Prompt Shields, under adaptive
    attacks. Treat this guardrail as a useful defense-in-depth signal, not a complete
    mitigation.

For more information, see:

- [Azure AI Content Safety: Prompt Shields (concept)](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/concepts/jailbreak-detection)
- [Quickstart: use Prompt Shields](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/quickstart-jailbreak)
- [Content Safety REST API reference (text jailbreak / Prompt Shields)](https://learn.microsoft.com/en-us/rest/api/contentsafety/text-operations/detect-text-jailbreak)
- [Defending Against Indirect Prompt Injection Attacks With Spotlighting (arXiv:2403.14720)](https://arxiv.org/abs/2403.14720)
- [Evaluating hosted prompt-injection detectors under adaptive attacks (arXiv:2504.11168)](https://arxiv.org/pdf/2504.11168)

## Supported Models

- `azure-prompt-shields`

## Constructor

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `endpoint` | `str | None` | No | `None` | Azure Content Safety endpoint URL. If ``None``, the value is read from the ``CONTENT_SAFETY_ENDPOINT`` environment variable. |
| `api_key` | `str | None` | No | `None` | Azure Content Safety API key. If ``None``, the value is read from the ``CONTENT_SAFETY_KEY`` environment variable. |

Initialize the Azure Prompt Shields guardrail.

## validate

Detect direct and indirect prompt-injection attacks via Azure Prompt Shields.

At least one of ``user_prompt`` or ``documents`` must be provided. The guardrail is
considered invalid (``valid=False``) if Azure flags an attack in the user prompt or
in **any** of the supplied documents.

**Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `user_prompt` | `str | None` | No | `None` | End-user prompt to scan for direct prompt-injection / jailbreak attempts. If ``None``, only documents are analyzed. |
| `documents` | `list[str] | None` | No | `None` | Auxiliary documents (e.g. retrieved context, tool outputs) to scan for indirect (data-borne) prompt-injection. If ``None``, only the user prompt is analyzed. |

**Returns:** `GuardrailOutput`
