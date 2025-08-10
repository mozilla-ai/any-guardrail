### Requirements

- Python 3.11 or newer

### Installation

Install with `pip`:

```bash
pip install any-guardrail
```

### Basic Usage

`AnyGuardrail` provides a seamless interface for interacting with the guardrail models. It allows you to see a list of all the supported guardrails, and to instantiate each supported guardrails. Here is a full example:

```python
from any_guardrail import AnyGuardrail, GuardrailName, GuardrailOutput
supported_models = AnyGuardrail.get_all_supported_model_ids() # This provides all supported model ids
supported_guardrails = GuardrailName # This provides all supported guardrail names
guardrail = AnyGuardrail.create_guardrail(model_id="deepset/deberta-v3-base-injection", guardrail_name=GuardrailName.DEEPSET)
result = guardrail.validate("All smiles from me!")
assert result.unsafe == False
```

### Troubleshooting

Some of the models on HuggingFace require extra permissions to use. To do this, you'll need to create a HuggingFace profile and manually go through the permissions. Then, you'll need to download the HuggingFace Hub and login. One way to do this is:

```bash
pip install --upgrade huggingface_hub

hf auth login
```

More information can be found here: [HuggingFace Hub](https://huggingface.co/docs/huggingface_hub/en/quick-start#login-command)
