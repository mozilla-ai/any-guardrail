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

# Author-published policies, rubrics & criteria

Where a *prompt template* is the scaffold a guardrail wraps around input, many guardrails also need
you to supply the **fill-in content** — the `policy`, `rubric`, or `criteria`. Rather than write
those from scratch, you can fetch the ones a guardrail's authors publish. These live in a separate,
import-free **content registry** with a per-kind API:

```python
from any_guardrail import AnyGuardrail, GuardrailName

# Discover what's available (no model loads)
AnyGuardrail.list_policies(GuardrailName.SHIELD_GEMMA)     # -> ['dangerous_content', 'harassment', ...]
AnyGuardrail.list_criteria(GuardrailName.GRANITE_GUARDIAN) # -> ['answer_relevance', 'harm', ...]
AnyGuardrail.list_rubrics(GuardrailName.PROMETHEUS)        # -> ['factual_validity', 'helpfulness', ...]

# Fetch a ready-to-use string and hand it straight to the guardrail
policy = AnyGuardrail.get_policy(GuardrailName.SHIELD_GEMMA, "dangerous_content")
guard = AnyGuardrail.create(GuardrailName.SHIELD_GEMMA, policy=policy)

# Judges take a rubric or criteria the same way
rubric = AnyGuardrail.get_rubric(GuardrailName.PROMETHEUS, "helpfulness")
criteria = AnyGuardrail.get_criteria(GuardrailName.GRANITE_GUARDIAN, "harm")
```

Registered today: **ShieldGemma** harm-type policies, **Granite Guardian** risk criteria,
**Flow Judge** preset criteria + rubrics, and **Prometheus** scoring rubrics. Content mirrored from
a live source (Granite Guardian, Flow-Judge) is drift-tested to stay byte-identical; the rest is
harvested verbatim from the model card / repo (each item carries a `source` URL and `provenance`).

Guardrails whose policy/rubric is genuinely **bring-your-own** with no canonical author-published
default (Selene, GLIDER, CompassJudger, DynaGuard, gpt-oss-safeguard, …) are intentionally not
registered — you supply your own. The full catalog is exported to
[`schemas/guardrail_content.json`](https://raw.githubusercontent.com/mozilla-ai/any-guardrail/main/schemas/guardrail_content.json).
