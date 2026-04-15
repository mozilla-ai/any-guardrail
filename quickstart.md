### Requirements

- Python 3.11 or newer

### Installation

You can install the bare bones library as follows (only [`any_guardrails.guardrails.any_llm.AnyLlm`] will be available):

```bash
pip install any-guardrail
```

Or you can install it with the required dependencies for different guardrails:

```bash
pip install any-guardrail[huggingface]
```

Refer to [pyproject.toml](https://github.com/mozilla-ai/any-guardrail/blob/main/pyproject.toml) for a list of the options available.

### Basic Usage

`AnyGuardrail` provides a seamless interface for interacting with the guardrail models. It allows you to see a list of all the supported guardrails, and to instantiate each supported guardrails. Here is a full example:

```python
from any_guardrail import AnyGuardrail, GuardrailName, GuardrailOutput

guardrail = AnyGuardrail.create(GuardrailName.DEEPSET)

result: GuardrailOutput = guardrail.validate("All smiles from me!")

assert result.valid
```

### Troubleshooting

Some of the models on HuggingFace require extra permissions to use. To do this, you'll need to create a HuggingFace profile and manually go through the permissions. Then, you'll need to download the HuggingFace Hub and login. One way to do this is:

```bash
pip install --upgrade huggingface_hub

hf auth login
```

More information can be found here: [HuggingFace Hub](https://huggingface.co/docs/huggingface_hub/en/quick-start#login-command)
