# Custom Blocklists with Azure Content Safety

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mozilla-ai/any-guardrail/blob/main/docs/cookbook/azure_blocklist_slang_filter.ipynb)

## Setup

Install the required packages and configure your Azure credentials.

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


for var in ["CONTENT_SAFETY_KEY", "CONTENT_SAFETY_ENDPOINT"]:
    ensure_env_var(var)
```

## Create a Blocklist

Initialize the guardrail and create a new blocklist. Here we're creating one for Gen Alpha slang terms.

```python
from any_guardrail import AnyGuardrail, GuardrailName

guardrail = AnyGuardrail.create(GuardrailName.AZURE_CONTENT_SAFETY)

blocklist_name = "GenAlphaSlang"
blocklist_description = "List of gen alpha words"

guardrail.create_or_update_blocklist(
    blocklist_name=blocklist_name,
    blocklist_description=blocklist_description,
)
```

##Add Terms to the Blocklist

Add the specific terms you want to filter. These can be individual words or phrases.

```python
blocklist_terms = [
    "Skibidi",
    "Rizz",
    "Sigma",
    "Gyatt",
    "Brain Rot",
    "Fanum Tax",
    "Ohio",
    "Mewing",
    "Aura",
    "Sigma",
    "Crash Out",
    "Delulu",
    "Glaze",
    "Mog",
    "Pookie",
    "Opp",
    "Slay",
]
guardrail.add_blocklist_items(blocklist_name=blocklist_name, blocklist_terms=blocklist_terms)
```

## Validate Text

Test the blocklist against sample text. The guardrail returns `valid=True` if no blocked terms are found, and `valid=False` with details about which terms were matched.

```python
# Pass
text = "Hello, how are you?"
result = guardrail.validate(text)
print(f"Text: {text} \nEvaluation result:{result} ")

# Fail - contains a term from the block list
text = "The startup pitch was all delulu with no solulu"
result = guardrail.validate(text)
print(f"Text: {text} \nEvaluation result:{result} ")
```

Below, test against a classic novel - *Anne of Green Gables* from Project Gutenberg. This demonstrates that literature from 1908 contains no Gen Alpha slang (as expected!).

```python
import gutenbergpy.textget

# Get a book by its Gutenberg ID (e.g., 45 for Anne of Green Gables)
# raw_book = gutenbergpy.textget.get_text_by_id(2701)
raw_book = gutenbergpy.textget.get_text_by_id(45)
# Strip headers and footers automatically
clean_book = gutenbergpy.textget.strip_headers(raw_book)
chunks = [clean_book[i : i + 7000] for i in range(0, len(clean_book), 7000)]
result = guardrail.validate(chunks[0])

print(f"Result: {result}")
```

## Next Steps

- Try adding your own terms to the blocklist
- You can use blocklists to filter competitor brand names, profanity, or domain-specific terms
- Combine blocklist filtering with other Azure Content Safety features (hate, violence, etc.)

For more information, see the [Azure Content Safety blocklist documentation](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/how-to/use-blocklist).
