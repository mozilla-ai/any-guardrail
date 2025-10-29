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


`any-guardrail` provides a unified interface for AI safety guardrails, for example, letting you detect toxic content, PII leakage, jailbreak attempts, and other risks in LLM inputs and outputs. Switch between different guardrail providers, both encoder-based (discriminative) and decoder-based (generative) models like Llama Guard and ShieldGemma, without changing your code.

Some guardrails are extremely customizable, which `any-guardrail` fully exposes. See the complete list of supported providers and customization examples in our [docs](https://mozilla-ai.github.io/any-guardrail/).

## Why any-guardrail?

- **Unified API**: Switch between evergrowing list of guardrail providers without code changes
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

## Documentation
Full guides at [docs link](https://mozilla-ai.github.io/any-guardrail/)

## Troubleshooting

Some of the models on HuggingFace require extra permissions to use. To do this, you'll need to create a HuggingFace profile and manually go through the permissions. Then, you'll need to download the HuggingFace Hub and login. One way to do this is:

```bash
pip install --upgrade huggingface_hub

hf auth login
```

More information can be found here: [HuggingFace Hub](https://huggingface.co/docs/huggingface_hub/en/quick-start#login-command)

## Contributing to `any-guardrail`

The guardrail space is ever growing. If there is a guardrail that you'd like us to support, please see our [CONTRIBUTING.md](CONTRIBUTING.md) for details.
