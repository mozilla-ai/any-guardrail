# DeepKeep

Screen input text using an existing DeepKeep deployment and a configured input
firewall. No additional dependency extra or local model weights are needed.

```python
from any_guardrail import AnyGuardrail, GuardrailName

guardrail = AnyGuardrail.create(
    GuardrailName.DEEPKEEP,
    api_key="your-deepkeep-api-key",
    endpoint="https://deepkeep.example.com",
    firewall_id="your-input-firewall-id",
)
result = AnyGuardrail.evaluate(GuardrailName.DEEPKEEP, guardrail, "Text to screen")
print(result.valid)
```

Supply the deployment base URL without the moderation API path. The firewall ID
is sent as DeepKeep's `model` field; it is not the model used for LLM inference.
Use a secret store for real credentials, not a hard-coded key.

This initial adapter checks text input only. A boolean `flagged: true` is always a
violation, including alert/redact/modify outcomes. It does not rewrite text or
inspect model output. Network failures, redirects, HTTP errors, and invalid
verdicts raise errors so the application can apply its unavailable policy.

Content logging defaults to false, TLS verification stays enabled, and the raw
response may contain sensitive text. Self-hosted Otari discovers DeepKeep from
the hosted-API metadata and generated parameter schema; its saved native
definitions require HTTPS and keep the API key encrypted.

::: any_guardrail.guardrails.deepkeep.deepkeep.Deepkeep
