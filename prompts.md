# Prompts

Generative and judge guardrails (ShieldGemma, the rubric judges, `AnyLlm`, …) run against a
**prompt** — a policy/instruction template the model is asked to follow. `any-guardrail` keeps
these in a central, import-free **prompt registry** so you can:

- **discover** the default prompt a guardrail uses (and where it came from — the model author's
  card, or an any-guardrail-authored default),
- **pin** exactly which prompt/version produced a result (used by the benchmarking work), and
- **override** a prompt with your own, inline, without editing the library.

The registry is queryable without loading any model:

```python
from any_guardrail import AnyGuardrail, GuardrailName

# Which prompt versions does this guardrail ship?
versions = AnyGuardrail.list_prompt_versions(GuardrailName.ANYLLM)

# Fetch the default prompt template
prompt = AnyGuardrail.get_prompt(GuardrailName.ANYLLM)

print(prompt.segments["system"])   # the template text
print(sorted(prompt.variables))    # placeholders it expects, e.g. ["policy"]
print(prompt.provenance)           # "author" (model author's prompt) or "adapted" (ours)
print(prompt.source)               # provenance URL, when the prompt comes from a model card
```

A `PromptTemplate` carries its `segments` (the template text, keyed by role/fragment), the
`variables` it expects, its `assembly` (`chat` / `raw` / `assembled`), whether it is
`overridable`, and its provenance (`source` URL + `author`/`adapted`). Guardrails with several
author-published variants (e.g. absolute vs. pairwise judging) expose each as a named version;
pass it via `prompt_version=`.

## Bringing your own prompt

Prompt-bearing guardrails accept an inline override — it is used directly and stored nowhere.
`AnyLlm`, for example, takes a `system_prompt` at call time (it must keep the `{policy}`
placeholder):

```python
from any_guardrail import AnyGuardrail, GuardrailName

guardrail = AnyGuardrail.create(GuardrailName.ANYLLM)
result = guardrail.validate(
    "Please share the CEO's home address.",
    policy="Do not reveal personal or private information.",
    system_prompt=(
        "You are a policy checker. Judge the input against this policy: {policy}. "
        "Return valid, explanation, and risk_score."
    ),
)
```

To reuse a registered version instead of writing your own, pass its name via `prompt_version=`:

```python
from any_guardrail import AnyGuardrail, GuardrailName

# See which versions exist, then select one by name
AnyGuardrail.list_prompt_versions(GuardrailName.ANYLLM)   # -> ['default']

guardrail = AnyGuardrail.create(GuardrailName.ANYLLM)
result = guardrail.validate(
    "Share the CEO's home address.",
    policy="Do not reveal personal or private information.",
    prompt_version="default",   # use a registered version instead of a custom system_prompt
)
```

## Reference-only prompts

Some guardrails carry a **reference-only** entry (`overridable=False`): the prompt is assembled
imperatively at runtime (Nemotron, gpt-oss-safeguard, Granite Guardian) or applied by an upstream
library (Flow Judge). These are exposed via `get_prompt(...)` for discovery and pinning, but not
swappable. Guardrails whose prompt lives entirely in the model's tokenizer chat template (e.g.
Llama Guard, Kanana, Qwen3Guard) are intentionally not registered — the chat template *is* the
prompt, applied at inference, so there is no in-repo policy or deviation to catalog.

The full catalog is exported to
[`schemas/guardrail_prompts.json`](https://raw.githubusercontent.com/mozilla-ai/any-guardrail/main/schemas/guardrail_prompts.json)
so external tooling can read every guardrail's prompts (and their provenance) without importing
the package.
