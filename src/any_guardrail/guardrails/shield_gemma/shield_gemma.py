from typing import ClassVar

from any_guardrail.base import GuardrailName, GuardrailOutput, StandardGuardrail
from any_guardrail.guardrails.utils import default
from any_guardrail.providers.base import StandardProvider
from any_guardrail.providers.huggingface import HuggingFaceProvider
from any_guardrail.registry import GUARDRAIL_METADATA
from any_guardrail.taxonomy import GuardrailMetadata
from any_guardrail.types import (
    AnyDict,
    GuardrailPreprocessOutput,
    StandardInferenceOutput,
    StandardPreprocessOutput,
)

SYSTEM_PROMPT_SHIELD_GEMMA = """You are a policy expert trying to help determine whether a user
prompt is in violation of the defined safety policies.

<start_of_turn>
Human Question: {user_prompt}
<end_of_turn>

Our safety principle is defined in the below:

{safety_policy}

Does the human question violate the above principle? Your answer must start
with 'Yes' or 'No'. And then walk through step by step to be sure we answer
correctly.
"""
DEFAULT_THRESHOLD: float = 0.5


class ShieldGemma(StandardGuardrail):
    """ShieldGemma — policy-conditioned safety classifier that judges a prompt against a user-supplied policy via Yes/No token logits (Google).

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

    Args:
        policy: The free-text safety principle the prompt is judged against (bring-your-own
            policy), inserted into the judgment template as the safety policy, e.g.
            ``"No Dangerous Content: The prompt shall not contain or seek generation of content
            that harms oneself or others ..."``.
        threshold: Decision threshold on the ``Yes`` (violation) probability. ``valid`` is
            ``score < threshold``. Defaults to ``0.5``.
        model_id: Optional HuggingFace model ID; must be one of ``SUPPORTED_MODELS``. Defaults to
            ``google/shieldgemma-2b``.
        provider: Optional pre-configured provider. Defaults to a ``HuggingFaceProvider`` loading
            the checkpoint as a causal LM.

    """

    SUPPORTED_MODELS: ClassVar = [
        "google/shieldgemma-2b",
        "google/shieldgemma-9b",
        "google/shieldgemma-27b",
    ]

    METADATA: ClassVar[GuardrailMetadata] = GUARDRAIL_METADATA[GuardrailName.SHIELD_GEMMA]

    def __init__(
        self,
        policy: str,
        threshold: float = DEFAULT_THRESHOLD,
        model_id: str | None = None,
        provider: StandardProvider | None = None,
    ) -> None:
        """Initialize the ShieldGemma guardrail.

        Args:
            policy: The free-text safety principle the prompt is judged against, inserted into
                ShieldGemma's judgment template as ``{safety_policy}``. Bring-your-own policy —
                e.g. ``"No Hate Speech: The prompt shall not contain or seek generation of content
                that targets identity or protected attributes ..."``.
            threshold: Decision threshold on the ``Yes`` (violation) probability. ``valid`` is
                ``score < threshold``; raise it to flag only higher-confidence violations, lower it
                to be stricter. Defaults to ``0.5``.
            model_id: Optional HuggingFace model ID; must be one of ``SUPPORTED_MODELS``. Defaults
                to ``google/shieldgemma-2b``.
            provider: Optional pre-configured provider. When ``None``, a ``HuggingFaceProvider`` is
                built targeting a causal LM (``AutoModelForCausalLM`` + ``AutoTokenizer``). A
                supplied ``HuggingFaceProvider`` is corrected to those classes at load time so the
                Yes/No logit head is available; any other provider is used as-is.

        """
        self.model_id = default(model_id, self.SUPPORTED_MODELS)
        self.policy = policy
        self.system_prompt = SYSTEM_PROMPT_SHIELD_GEMMA
        self.threshold = threshold
        # Lazy-import transformers so importing ShieldGemma does not require
        # the huggingface extra. A caller could supply a non-HF provider that
        # returns torch tensors from infer() — the transformers classes are
        # only needed when the provider is a HuggingFaceProvider (or when we
        # construct one ourselves).
        load_kwargs: AnyDict = {}
        if provider is not None:
            self.provider = provider
            if isinstance(self.provider, HuggingFaceProvider):
                # ShieldGemma is a causal LM. A default-constructed
                # HuggingFaceProvider targets AutoModelForSequenceClassification,
                # which would load Gemma2ForSequenceClassification — the
                # checkpoint has no classification head, so score.weight is
                # randomly initialized and _post_processing later crashes on
                # 2D logits. Enforce the right classes for this load.
                from transformers import AutoModelForCausalLM, AutoTokenizer

                load_kwargs = {"model_class": AutoModelForCausalLM, "tokenizer_class": AutoTokenizer}
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.provider = HuggingFaceProvider(model_class=AutoModelForCausalLM, tokenizer_class=AutoTokenizer)
        self.provider.load_model(self.model_id, **load_kwargs)

    def _pre_processing(self, input_text: str) -> StandardPreprocessOutput:
        formatted_prompt = self.system_prompt.format(user_prompt=input_text, safety_policy=self.policy)
        tokenized = self.provider.tokenizer(formatted_prompt, return_tensors="pt")  # type: ignore[attr-defined]
        device = getattr(self.provider, "device", None)
        if device is not None:
            tokenized = tokenized.to(device)
        return GuardrailPreprocessOutput(data=tokenized)

    def _inference(self, model_inputs: StandardPreprocessOutput) -> StandardInferenceOutput:
        return self.provider.infer(model_inputs)

    def _post_processing(self, model_outputs: StandardInferenceOutput) -> GuardrailOutput:
        # Lazy-import torch so importing ShieldGemma does not require the
        # huggingface extra at module load. ShieldGemma still needs torch
        # at validate() time to slice the causal-LM logits, but a user who
        # never calls validate() can import the class freely.
        from torch.nn.functional import softmax

        logits = model_outputs.data["logits"]
        vocab = self.provider.tokenizer.get_vocab()  # type: ignore[attr-defined]
        selected_logits = logits[0, -1, [vocab["Yes"], vocab["No"]]]
        probabilities = softmax(selected_logits, dim=0)
        score = probabilities[0].item()
        return GuardrailOutput(valid=score < self.threshold, score=score)
