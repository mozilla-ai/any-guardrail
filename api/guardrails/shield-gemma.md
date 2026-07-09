# ShieldGemma

ShieldGemma — policy-conditioned safety classifier that judges a prompt against a user-supplied policy via Yes/No token logits (Google).

ShieldGemma is Google's Gemma-2-based content-safety classifier. Rather than a fixed taxonomy,
it is conditioned at construction on a free-text ``policy`` (a safety principle). Each call
inserts the user prompt and the policy into ShieldGemma's judgment template — which asks
whether the prompt violates the principle and requires the answer to start with ``Yes`` or
``No`` — runs the causal LM, and reads the logits of the ``Yes`` / ``No`` vocabulary tokens at
the final position, softmaxing them into a violation probability.

Verdict mapping onto ``GuardrailOutput``:

- ``score`` is the probability mass on ``Yes`` (the policy is violated) — the canonical risk
  axis, higher = riskier.
- ``valid`` is ``score < threshold`` (default ``0.5``): the prompt passes when the violation
  probability stays below the threshold.
- No ``categories``, ``spans``, or ``explanation`` are produced.

Expected input: a single ``input_text`` prompt string, judged against the constructor's
``policy``. This is prompt-only moderation; there is no response or RAG-context channel. Only
the text classifier is wrapped — the ShieldGemma image classifier is not supported.

The models are gated on HuggingFace under the Gemma Terms of Use.

For more information, see:

- [ShieldGemma collection (Google)](https://huggingface.co/collections/google/shieldgemma-67d130ef8da6af884072a789)
- [google/shieldgemma-2b](https://huggingface.co/google/shieldgemma-2b)
- [google/shieldgemma-9b](https://huggingface.co/google/shieldgemma-9b)
- [google/shieldgemma-27b](https://huggingface.co/google/shieldgemma-27b)

## Supported Models

- `google/shieldgemma-2b`
- `google/shieldgemma-9b`
- `google/shieldgemma-27b`

## Constructor

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `policy` | `str` | Yes | — | The free-text safety principle the prompt is judged against, inserted into ShieldGemma's judgment template as ``{safety_policy}``. Bring-your-own policy — e.g. ``"No Hate Speech: The prompt shall not contain or seek generation of content that targets identity or protected attributes ..."``. |
| `threshold` | `float` | No | `0.5` | Decision threshold on the ``Yes`` (violation) probability. ``valid`` is ``score < threshold``; raise it to flag only higher-confidence violations, lower it to be stricter. Defaults to ``0.5``. |
| `model_id` | `str | None` | No | `None` | Optional HuggingFace model ID; must be one of ``SUPPORTED_MODELS``. Defaults to ``google/shieldgemma-2b``. |
| `provider` | `Optional[Provider[dict[str, Any], dict[str, Any]]]` | No | `None` | Optional pre-configured provider. When ``None``, a ``HuggingFaceProvider`` is built targeting a causal LM (``AutoModelForCausalLM`` + ``AutoTokenizer``). A supplied ``HuggingFaceProvider`` is corrected to those classes at load time so the Yes/No logit head is available; any other provider is used as-is. |

Initialize the ShieldGemma guardrail.

## validate

Default validation pipeline: preprocess -> inference -> postprocess.

**Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `input_text` | `str | list[str]` | Yes | — | The text to validate. If a list is supplied, each item is validated and a list of GuardrailOutputs is returned in the same order. Subclasses can override ``_validate_batch`` to enable true batched inference; the default iterates over inputs. |

**Returns:** `GuardrailOutput | list[GuardrailOutput]`
