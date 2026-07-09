# DuoGuard

DuoGuard — multilingual multi-label safety classifier scoring text across 12 harm categories including jailbreak prompts.

DuoGuard is a compact (0.5B-1.5B) classifier built on Qwen 2.5 and Llama 3.2 backbones,
trained with a two-player reinforcement-learning framework in which a generator and the
guardrail co-evolve to synthesize multilingual safety data. It emits an independent
probability (sigmoid) for each of the 12 categories in ``DUOGUARD_CATEGORIES`` — the
MLCommons-style hazard taxonomy (violent crimes, non-violent crimes, sex-related crimes,
child sexual exploitation, specialized advice, privacy, intellectual property,
indiscriminate weapons, hate, suicide and self-harm, sexual content) plus a
jailbreak-prompt category.

The models are fine-tuned primarily for English, French, German, and Spanish, with broader
coverage inherited from the Qwen 2.5 / Llama 3.2 base models.

Verdict mapping onto ``GuardrailOutput``:

- ``categories`` carries all 12 categories, each with its sigmoid probability and a
  ``triggered`` flag (probability strictly above ``threshold``).
- ``valid`` is ``True`` only when no category is triggered.
- ``score`` (canonical risk: higher = riskier) is the maximum category probability.

Expected inputs: a single text string, or a ``list[str]`` for batched classification (the
inherited ``ThreeStageGuardrail.validate`` handles list input). It screens a single body
of text; it does not take a separate prompt / response pair.

For more information, see:

- [DuoGuard model collection](https://huggingface.co/collections/DuoGuard/duoguard-models-67a29ad8bd579a404e504d21).
- [DuoGuard-0.5B model card](https://huggingface.co/DuoGuard/DuoGuard-0.5B) (default).
- [DuoGuard-1B-Llama-3.2-transfer model card](https://huggingface.co/DuoGuard/DuoGuard-1B-Llama-3.2-transfer).
- [DuoGuard-1.5B-transfer model card](https://huggingface.co/DuoGuard/DuoGuard-1.5B-transfer).
- [DuoGuard: A Two-Player RL-Driven Framework for Multilingual LLM Guardrails (arXiv:2502.05163)](https://arxiv.org/abs/2502.05163).

## Supported Models

- `DuoGuard/DuoGuard-0.5B`
- `DuoGuard/DuoGuard-1B-Llama-3.2-transfer`
- `DuoGuard/DuoGuard-1.5B-transfer`

## Constructor

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model_id` | `str | None` | No | `None` | Optional HuggingFace model ID. Must be one of ``SUPPORTED_MODELS``; defaults to ``DuoGuard/DuoGuard-0.5B``. The larger ``DuoGuard/DuoGuard-1B-Llama-3.2-transfer`` and ``DuoGuard/DuoGuard-1.5B-transfer`` variants trade latency for accuracy. |
| `threshold` | `float` | No | `0.5` | Per-category probability strictly above which that category is flagged (and the text becomes invalid). Defaults to 0.5, from the model card. |
| `provider` | `Provider[dict[str, Any], dict[str, Any]] | None` | No | `None` | Optional pre-configured provider. If ``None``, a default ``HuggingFaceProvider`` is built with ``multi_label=True`` and the matching base tokenizer (see ``MODELS_TO_TOKENIZER``), then the model is loaded eagerly. A pad token is set from the EOS token because the Qwen-family tokenizers ship without one. |

Initialize the DuoGuard guardrail.

## validate

Default validation pipeline: preprocess -> inference -> postprocess.

**Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `input_text` | `str | list[str]` | Yes | — | The text to validate. If a list is supplied, each item is validated and a list of GuardrailOutputs is returned in the same order. Subclasses can override ``_validate_batch`` to enable true batched inference; the default iterates over inputs. |

**Returns:** `GuardrailOutput | list[GuardrailOutput]`
