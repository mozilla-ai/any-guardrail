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

## [Documentation](https://mozilla-ai.github.io/any-guardrail/)

## Motivation

LLM Guardrail and Judge models can be seen as a combination of an LLM + some classification function. This leads to some churn when one wants to experiment with guardrails to see which fits their use case or to compare guardrails. `any-guardrail` is built to provide a seamless interface to many guardrail models, both encoder (discriminative) and decoder (generative), to easily swap them out for downstream use cases and research.

## Our Approach

`any-guardrail` is meant to provide the minimum amount of access necessary to implement the guardrails in your pipeline. We do this by providing taking care of the loading and instantiation of a model or pipeline in the backend, and providing a `validate` function to classify.

Some guardrails are extremely customizable and we allow for that customization as well. We recommend reading our [docs](https://mozilla-ai.github.io/any-guardrail/) to see how to build more customized use cases.

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

guardrail = AnyGuardrail.create(GuardrailName.DEEPSET)

result: GuardrailOutput = guardrail.validate("All smiles from me!")

assert result.valid
```

## Troubleshooting

Some of the models on HuggingFace require extra permissions to use. To do this, you'll need to create a HuggingFace profile and manually go through the permissions. Then, you'll need to download the HuggingFace Hub and login. One way to do this is:

```bash
pip install --upgrade huggingface_hub

hf auth login
```

More information can be found here: [HuggingFace Hub](https://huggingface.co/docs/huggingface_hub/en/quick-start#login-command)

## Contributing to `any-guardrail`

The guardrail space is ever growing. If there is a guardrail that you'd like us to support, please see our [CONTRIBUTING.md](CONTRIBUTING.md) for details.
