## Quickstart

### Requirements

- Python 3.11 or newer
- For guardrails that need permission granted on HuggingFace, make sure to get a HuggingFace access token as well. Then log into [HuggingFace Hub](https://huggingface.co/docs/huggingface_hub/en/quick-start#login-command)

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

`GuardrailFactory` provides a seamless interface for interacting with the guardrail models. It allows you to see a list of all the supported guardrails, and to instantiate each supported guardrails. Here is a full example:

```python
from any_guardrail import GuardrailFactory
factory = GuardrailFactory()
supported_guardrails = factory.list_all_supported_guardrails() # This will out a list of all guardrail identifiers
guardrail = factory.create_guardrail(support_guardrails[0])  # will create Deepset's deberta prompt injection defense model
result = guardrail.safety_review("All smiles from me!")
assert result.unsafe == False
```
