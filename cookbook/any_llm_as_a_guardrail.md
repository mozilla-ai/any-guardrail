# Using Any LLM as a Guardrail

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mozilla-ai/any-guardrail/blob/main/docs/cookbook/any_llm_as_a_guardrail.ipynb)

## Install dependencies

```python
import nest_asyncio

nest_asyncio.apply()
```

We will be using a model from `openai` by default, but you can check
the different providers supported in `any-llm`:

https://mozilla-ai.github.io/any-llm/providers/

```python
import os
from getpass import getpass

if "OPENAI_API_KEY" not in os.environ:
    print("OPENAI_API_KEY not found in environment!")
    api_key = getpass("Please enter your OPENAI_API_KEY: ")
    os.environ["OPENAI_API_KEY"] = api_key
    print("OPENAI_API_KEY set for this session!")
else:
    print("OPENAI_API_KEY found in environment.")
```

## Create the guardrail

```python
from any_guardrail import AnyGuardrail, GuardrailName
```

```python
guardrail = AnyGuardrail.create(GuardrailName.ANYLLM)
```

## Try it with different models / policies / inputs

```python
MODEL_ID = "openai/gpt-5-nano"

POLICY = """
You hate Mondays.
You must reject any request related with planning activities on Mondays.
"""
```

```python
guardrail.validate("Can you suggest me some restaurants for lunch on Monday?", policy=POLICY, model_id=MODEL_ID)
```

```python
guardrail.validate("Can you suggest me some restaurants for lunch on Friday?", policy=POLICY, model_id=MODEL_ID)
```
