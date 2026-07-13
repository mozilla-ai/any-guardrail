<p align="center">
  <picture>
    <img src="docs/images/any-guardrail-favicon.png" width="20%" alt="Project logo"/>
  </picture>
</p>

<div align="center">

# any-guardrail

[![Docs](https://github.com/mozilla-ai/any-guardrail/actions/workflows/docs.yaml/badge.svg)](https://github.com/mozilla-ai/any-guardrail/actions/workflows/docs.yaml/)
[![Linting](https://github.com/mozilla-ai/any-guardrail/actions/workflows/lint.yaml/badge.svg)](https://github.com/mozilla-ai/any-guardrail/actions/workflows/lint.yaml/)
[![Unit Tests](https://github.com/mozilla-ai/any-guardrail/actions/workflows/tests-unit.yaml/badge.svg)](https://github.com/mozilla-ai/any-guardrail/actions/workflows/tests-unit.yaml/)
[![Integration Tests](https://github.com/mozilla-ai/any-guardrail/actions/workflows/tests-integration.yaml/badge.svg)](https://github.com/mozilla-ai/any-guardrail/actions/workflows/tests-integration.yaml/)

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)
[![PyPI](https://img.shields.io/pypi/v/any-guardrail)](https://pypi.org/project/any-guardrail/)
<a href="https://discord.gg/4gf3zXrQUc">
    <img src="https://img.shields.io/static/v1?label=Chat%20on&message=Discord&color=blue&logo=Discord&style=flat-square" alt="Discord">
</a>

A single interface to use different guardrail models.

</div>


`any-guardrail` provides a unified interface for AI safety guardrails, for example, letting you detect toxic content, jailbreak attempts, and other risks in LLM inputs and outputs. Switch between different guardrail providers, both encoder-based (discriminative) and decoder-based (generative) models like Llama Guard and ShieldGemma, without changing your code.

Some guardrails are extremely customizable, which `any-guardrail` fully exposes. See the complete list of supported providers and customization examples in our [docs](https://docs.mozilla.ai/any-guardrail).

## Why any-guardrail?

- **Unified API**: Switch between evergrowing list of guardrail providers
- **Production-ready**: Built for real-world LLM applications
- **Flexible**: Use encoder-based (fast) or decoder-based (customizable) models

## Quickstart

### Requirements

- Python 3.11 or newer

### Installation

Install with `pip`:

```bash
pip install any-guardrail
```

### Basic Usage

`AnyGuardrail` provides a seamless interface for interacting with the guardrail models. It allows you to see a list of all the supported guardrails, and to instantiate each supported guardrail. Here is a full example:

```python
from any_guardrail import AnyGuardrail, GuardrailName, GuardrailOutput

# Initialize guardrail
guardrail = AnyGuardrail.create(GuardrailName.DEEPSET)

# Validate input before sending to your LLM
result: GuardrailOutput = guardrail.validate("How do I hack into a system?")

if not result.valid:
    print(f"Blocked: {result.explanation}")
else:
    # Safe to proceed with LLM call
    response = your_llm(user_input)
```

Every guardrail returns the same `GuardrailOutput` shape, so you can swap models without changing application code:

```python
result.valid       # bool verdict — True means the content passed
result.score       # risk score in ~[0, 1], higher = more likely violating (when available)
result.categories  # per-category results: CategoryResult(name, description, triggered, score, severity)
result.explanation # human-readable rationale (judge reasoning, raw generation)
result.action      # provider-recommended action (e.g. "block"), advisory; None if none
result.usage       # provenance: model_id, latency_ms, token counts
result.extra       # guardrail-specific structured extras; result.raw holds the backend payload

flagged = [c.name for c in result.categories if c.triggered]
```

A machine-readable [JSON Schema](schemas/guardrail_output.schema.json) for this output is published in the repo (generated from the Pydantic models). Reference it at the stable raw URL, pinning a release tag for a specific version:

```
https://raw.githubusercontent.com/mozilla-ai/any-guardrail/main/schemas/guardrail_output.schema.json
```

### Prompts, policies, rubrics & criteria

Generative and judge guardrails run against a **prompt template** and often need a **policy**,
**rubric**, or **criteria**. any-guardrail lets you discover and override the default prompt, and
fetch author-published policies/rubrics/criteria — all without loading a model:

```python
from any_guardrail import AnyGuardrail, GuardrailName

# Inspect a guardrail's default prompt template
AnyGuardrail.get_prompt(GuardrailName.SELENE).segments["user"]

# Fetch a ready-made author-published policy and use it directly
policy = AnyGuardrail.get_policy(GuardrailName.SHIELD_GEMMA, "dangerous_content")
guard = AnyGuardrail.create(GuardrailName.SHIELD_GEMMA, policy=policy)
```

See the [Prompts & content guide](docs/prompts.md). The catalogs are exported to
[`schemas/guardrail_prompts.json`](schemas/guardrail_prompts.json) and
[`schemas/guardrail_content.json`](schemas/guardrail_content.json).

## Documentation
Full guides at [docs link](https://docs.mozilla.ai/any-guardrail)

## Troubleshooting

Some of the models on HuggingFace require extra permissions to use. To do this, you'll need to create a HuggingFace profile and manually go through the permissions. Then, you'll need to download the HuggingFace Hub and login. One way to do this is:

```bash
pip install --upgrade huggingface_hub

hf auth login
```

More information can be found here: [HuggingFace Hub](https://huggingface.co/docs/huggingface_hub/en/quick-start#login-command)

## Contributing to `any-guardrail`

The guardrail space is ever growing. If there is a guardrail that you'd like us to support, please see our [CONTRIBUTING.md](CONTRIBUTING.md) for details.
