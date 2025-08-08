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
supported_guardrails = AnyGuardrail.list_all_supported_guardrails() # This will out a list of all guardrail identifiers
guardrail = AnyGuardrail.create_guardrail(model_id="google/shieldgemma-2b", guardrail_name=GuardrailName.SHIELD_GEMMA)
result: GuardrailOutput = guardrail.safety_review("All smiles from me!")
assert result.unsafe == False
```

### Troubleshooting

Some of the models on HuggingFace require extra permissions to use. To do this, you'll need to create a HuggingFace profile and manually go through the permissions. Then, you'll need to download the HuggingFace Hub and login. One way to do this is:

```bash
pip install --upgrade huggingface_hub

hf auth login
```

More information can be found here: [HuggingFace Hub](https://huggingface.co/docs/huggingface_hub/en/quick-start#login-command)
