

# any-guardrail

[![Docs](https://github.com/mozilla-ai/any-guardrail/actions/workflows/docs.yaml/badge.svg)](https://github.com/mozilla-ai/any-guardrail/actions/workflows/docs.yaml/)
[![Linting](https://github.com/mozilla-ai/any-guardrail/actions/workflows/lint.yaml/badge.svg)](https://github.com/mozilla-ai/any-guardrail/actions/workflows/lint.yaml/)
[![Unit Tests](https://github.com/mozilla-ai/any-guardrail/actions/workflows/tests-unit.yaml/badge.svg)](https://github.com/mozilla-ai/any-guardrail/actions/workflows/tests-unit.yaml/)
[![Integration Tests](https://github.com/mozilla-ai/any-guardrail/actions/workflows/tests-integration.yaml/badge.svg)](https://github.com/mozilla-ai/any-guardrail/actions/workflows/tests-integration.yaml/)

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)
[![PyPI](https://img.shields.io/pypi/v/any-guardrail-sdk)](https://pypi.org/project/any-guardrail-sdk/)
<a href="https://discord.gg/4gf3zXrQUc">
    <img src="https://img.shields.io/static/v1?label=Chat%20on&message=Discord&color=blue&logo=Discord&style=flat-square" alt="Discord">
</a>

A single interface to use different guardrail models.

</div>

## [Documentation](https://mozilla-ai.github.io/any-guardrail/)

## Motivation

LLM Guardrail and Judge models can be seen as a combination of an LLM + some classification function. This leads to some churn when one wants to experiment with guardrails to see which fits their use case or to compare guardrails. `any-guardrail` is built to provide a seamless interface to many guardrail models, both encoder (discriminative) and decoder (generative), to easily swap them out for downstream use cases and research.

## Our Approach

`any-guardrail` is meant to provide the minimum amount of access necessary to implement the guardrails in your pipeline. We do this by providing taking care of the loading and instantiation of a model or pipeline in the backend, and providing a `safety_review` function to classify.

Some guardrails are extremely customizable and we allow for that customization as well. We recommend reading our [docs](https://mozilla-ai.github.io/any-guardrail/) to see how to build more customized use cases.

## Quickstart

### Requirements

- Python 3.11 or newer
- For guardrails that need permission granted on HuggingFace, make sure to get a HuggingFace access token as well. The log into [HuggingFace Hub](https://huggingface.co/docs/huggingface_hub/en/quick-start#login-command)

### Installation

#### Using `pip`

To install, you can use `pip`:

```bash
pip install any-guardrail
```

If you plan to use HuggingFace models that require extra permissions, please log into the HuggingFace Hub:

```bash
pip install --upgrade huggingface_hub

hf auth login
```

Make sure to agree to the terms and conditions of the model you are trying to access as well.

#### Using `uv`

If you would like to use the package from the GitHub repo, you can do use `uv`:

```bash
uv sync

uv venv

source .venv/bin/activate
```

### Basic Usage

The best way to use our package is to instantiate the `GuardrailFactory` first.

```python
from any_guardrail.api import GuardrailFactory

factory = GuardrailFactory
```

`GuardrailFactory` has a couple of functions to make using the package easier.
First, you can use:

```python
factory.list_all_supported_models()
```

This will output all of our support model names that you can pass into our `GuardrailFactory`. Here is how to do it:

```python
guardrail = factory.create_guardrail("model/identifier/or/path")
```

This will now give you the designated guardrail, which you can then use to review text output.

```python
guardrail.safety_review("<text I want to review>")
```

We have an example notebook to help as well in `examples/Testing.ipynb`.

## Advanced Usage

If a guardrail is not available, fork this repo to add it and then issue a pull request. Please use the following steps:

### Step 1: Create a `Guardrail` class

We have an abstract `Guardrail` class that has the minimum api required to create a new guardrail. To do so, implement the following:

```python
class YourGuardrail(Guardrail):
    def __init__(self, ...):
        super().__init__(model_identifier)
        self.guardrail = _model_instantiation(model_identifier, ...)

    def safety_review(...):
        # Your implementation for reviewing text

    def _model_instantiation(...):
        # Your implementation for instantiating a model
```

For more detailed examples, we recommend looking through the `guardrails` directory.

### Step 2: Add your model to the `model_registry.py`

Now that you have created `YourGuardrail`, you need add a model identifier to help the `GuardrailFactory` identify your guardrail. It will look something like this:

```python

model_registry= {
    "already/implemented/guardrail": SomeGuardrail,
    ...
    "your/guardrail/identifier": YourGuardrail
}
```

From there, you should be all set!
