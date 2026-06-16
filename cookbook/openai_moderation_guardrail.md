# OpenAI Moderation Guardrail Usage

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mozilla-ai/any-guardrail/blob/main/docs/cookbook/openai_moderation_guardrail.ipynb)

```python
import os
from getpass import getpass


def ensure_env_var(name: str) -> None:
    """Prompt for an environment variable if not already set."""
    if name not in os.environ:
        print(f"{name} not found in environment!")
        value = getpass(f"Please enter your {name}: ")
        os.environ[name] = value
        print(f"{name} set for this session!")
    else:
        print(f"{name} found in environment.")


for var in ["OPENAI_API_KEY"]:
    ensure_env_var(var)
```

# Basic Usage

```python
from any_guardrail import AnyGuardrail, GuardrailName

guardrail = AnyGuardrail.create(GuardrailName.OPENAI_MODERATION)
```

```python
output = guardrail.validate("What's the weather like in Paris today?")
print(f"valid={output.valid}, score={output.score}")
```

```python
output = guardrail.validate("I will hunt you down and break every bone in your body, you worthless piece of trash.")
print(f"valid={output.valid}, score={output.score}")
```

Output should look like:
```
valid=False, score=0.953...
```

- `valid` is `False` when OpenAI flags the content **or** when the maximum per-category score exceeds the `threshold` (default `0.5`).
- `score` is the maximum category score.
- `categories` is the full per-category breakdown, each with a `score` and a `triggered` flag:

```python
sorted([(c.name, c.score) for c in output.categories], key=lambda kv: -(kv[1] or 0))[:3]
```

# Advanced Usage

For more information, please see our [docs](https://mozilla-ai.github.io/any-guardrail/api/guardrails/openai-moderation/).

## Customizing the threshold

OpenAI's `flagged` verdict is conservative. If you want to block borderline content, lower the `threshold`: any category score above it makes the result invalid even when OpenAI did not flag it.

```python
strict_guardrail = AnyGuardrail.create(GuardrailName.OPENAI_MODERATION, threshold=0.1)

output = strict_guardrail.validate("mildly edgy text that OpenAI itself would not flag")
print(f"valid={output.valid}, score={output.score}")
```

## Pinning the model version

By default the guardrail uses `omni-moderation-latest` (a GPT-4o-derived multimodal classifier). You can pin a dated snapshot for reproducibility:

```python
pinned_guardrail = AnyGuardrail.create(
    GuardrailName.OPENAI_MODERATION,
    model_id="omni-moderation-2024-09-26",
)
```
