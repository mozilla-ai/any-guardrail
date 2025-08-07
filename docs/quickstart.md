## Quickstart

### Requirements

- Python 3.11 or newer
- For guardrails that need permission granted on HuggingFace, make sure to get a HuggingFace access token as well. The log into [HuggingFace Hub](https://huggingface.co/docs/huggingface_hub/en/quick-start#login-command)

### Installation

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

### Basic Usage

The best way to use our package is to instantiate the `GuardrailFactory`. It provides a seamless interface for interacting with the guardrail models.

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
