# LlamaGuard

Wrapper class for Llama Guard 3 & 4 implementations.

For more information about the implementations about either off topic model, please see the below model cards:

- [Meta Llama Guard 3 Docs](https://www.llama.com/docs/model-cards-and-prompt-formats/llama-guard-3/)
- [HuggingFace Llama Guard 3 Docs](https://huggingface.co/meta-llama/Llama-Guard-3-1B)
- [Meta Llama Guard 4 Docs](https://www.llama.com/docs/model-cards-and-prompt-formats/llama-guard-4/)
- [HuggingFace Llama Guard 4 Docs](https://huggingface.co/meta-llama/Llama-Guard-4-12B)

## Supported Models

- `meta-llama/Llama-Guard-3-1B`
- `meta-llama/Llama-Guard-3-8B`
- `meta-llama/Llama-Guard-4-12B`

## Constructor

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `model_id` | `str | None` | No | `None` |
| `provider` | `Optional[Provider[dict[str, Any], dict[str, Any]]]` | No | `None` |

Llama guard model. Either Llama Guard 3 or 4 depending on the model id. Defaults to Llama Guard 3.

## validate

Default validation pipeline: preprocess -> inference -> postprocess.

Args:
    input_text: The text to validate.
    **kwargs: Additional arguments passed to preprocessing (e.g., output_text, comparison_text).

Returns:
    GuardrailOutput with validation results.

Note:
    Subclasses can override this method to customize the signature or add validation logic.

**Parameters**

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `input_text` | `str` | Yes | — |

**Returns:** `GuardrailOutput`
