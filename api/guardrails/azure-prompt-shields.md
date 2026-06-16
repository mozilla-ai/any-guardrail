# AzurePromptShields

Guardrail wrapping Azure AI Content Safety's Prompt Shields feature.

Prompt Shields is a service from Azure AI Content Safety that detects prompt-injection
and jailbreak attacks against LLM applications. It supports two attack surfaces:

- **Direct attacks (user_prompt)**: malicious instructions in end-user input attempting
  to override the system prompt, exfiltrate sensitive info, or otherwise jailbreak
  the model.
- **Indirect attacks (documents)**: data-borne prompt injection embedded inside
  retrieved documents, tool outputs, or other context fed to the model. Microsoft
  Research's [Spotlighting](https://arxiv.org/abs/2403.14720) technique (Hines et al., 2024)
  is the published basis for this indirect-attack detection.

Concept documentation:
[Azure AI Content Safety: Prompt Shields](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/concepts/jailbreak-detection).
Quickstart:
[Use Prompt Shields](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/quickstart-jailbreak).

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

Args:
    endpoint: Azure Content Safety endpoint URL. Falls back to ``CONTENT_SAFETY_ENDPOINT``
        env var.
    api_key: Azure Content Safety API key. Falls back to ``CONTENT_SAFETY_KEY`` env var.

## Supported Models

- `azure-prompt-shields`

## Constructor

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `endpoint` | `str | None` | No | `None` |
| `api_key` | `str | None` | No | `None` |

Initialize the Azure Prompt Shields guardrail.

## validate

Detect direct and indirect prompt-injection attacks via Azure Prompt Shields.

At least one of ``user_prompt`` or ``documents`` must be provided. The guardrail is
considered invalid (``valid=False``) if Azure flags an attack in the user prompt or
in **any** of the supplied documents.

**Parameters**

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `user_prompt` | `str | None` | No | `None` |
| `documents` | `list[str] | None` | No | `None` |

**Returns:** `GuardrailOutput`
