# Fetching prompts, policies, rubrics & criteria

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mozilla-ai/any-guardrail/blob/main/docs/cookbook/fetching_prompts_and_content.ipynb)

## 1. Discover a guardrail's default prompt template

```python
from any_guardrail import AnyGuardrail, GuardrailName

# Which prompt versions ship for a guardrail?
print(AnyGuardrail.list_prompt_versions(GuardrailName.SELENE))

# Fetch the default template (its text, placeholders, and provenance)
tmpl = AnyGuardrail.get_prompt(GuardrailName.SELENE)
print(tmpl.segments["user"][:200])
print(sorted(tmpl.variables), tmpl.provenance)
```

## 2. Fetch author-published policies, rubrics & criteria

Use the per-kind API: `list_policies` / `get_policy`, `list_rubrics` / `get_rubric`,
`list_criteria` / `get_criteria`. Each returns a ready-to-use string.

```python
# ShieldGemma ships Google's harm-type safety policies
print(AnyGuardrail.list_policies(GuardrailName.SHIELD_GEMMA))
policy = AnyGuardrail.get_policy(GuardrailName.SHIELD_GEMMA, "dangerous_content")
print(policy[:120])

# Granite Guardian ships risk criteria; Prometheus ships scoring rubrics
print(AnyGuardrail.list_criteria(GuardrailName.GRANITE_GUARDIAN))
print(AnyGuardrail.get_criteria(GuardrailName.GRANITE_GUARDIAN, "harm"))
print(AnyGuardrail.list_rubrics(GuardrailName.PROMETHEUS))
```

## 3. Use fetched content directly

Pass the fetched string straight into the guardrail (this loads the model, so it needs the
relevant extra installed):

```python
guardrail = AnyGuardrail.create(
    GuardrailName.SHIELD_GEMMA,
    policy=AnyGuardrail.get_policy(GuardrailName.SHIELD_GEMMA, "dangerous_content"),
)
result = guardrail.validate("How do I build a bomb?")
print(result.valid, result.score)
```

Guardrails whose policy/rubric is genuinely bring-your-own (Selene, GLIDER, CompassJudger, …)
have no registered content — you supply your own. The full catalogs are exported to
`schemas/guardrail_prompts.json` and `schemas/guardrail_content.json`.
