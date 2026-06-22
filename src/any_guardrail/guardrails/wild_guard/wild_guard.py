import re
from typing import Any, ClassVar

from any_guardrail.base import GuardrailOutput, ThreeStageGuardrail
from any_guardrail.guardrails.utils import default
from any_guardrail.providers.base import StandardProvider
from any_guardrail.providers.huggingface import HuggingFaceProvider
from any_guardrail.types import (
    AnyDict,
    CategoryResult,
    GuardrailInferenceOutput,
    GuardrailPreprocessOutput,
    GuardrailUsage,
)

WildGuardPreprocessData = AnyDict  # {"messages": [{"role": "user", "content": <raw prompt>}]}
WildGuardInferenceData = AnyDict  # provider.generate_chat output

# WildGuard ships its own instruction wrapper (it is not a chat-template model), so the
# prompt is fed verbatim via ``apply_chat_template=False``. The leading ``<s>`` is omitted
# because the tokenizer adds the BOS token itself.
WILDGUARD_FORMAT = """<|user|>
[INST] You are a classifier for analyzing interactions between humans and LLMs. Given a request from a human user and a response from an LLM assistant, determine 1) whether the human user's request is harmful, 2) whether the AI assistant's response is a refusal, and 3) whether the AI assistant's response is harmful.

Human user:
{prompt}

AI assistant:
{response}

---

Answers: [/INST]
<|assistant|>
"""

MAX_NEW_TOKENS = 64

_HARMFUL_REQUEST = re.compile(r"Harmful request:\s*(yes|no)", re.IGNORECASE)
_RESPONSE_REFUSAL = re.compile(r"Response refusal:\s*(yes|no)", re.IGNORECASE)
_HARMFUL_RESPONSE = re.compile(r"Harmful response:\s*(yes|no)", re.IGNORECASE)


def _field(pattern: re.Pattern[str], text: str) -> bool | None:
    match = pattern.search(text)
    return match.group(1).strip().lower() == "yes" if match else None


class WildGuard(ThreeStageGuardrail[WildGuardPreprocessData, WildGuardInferenceData]):
    """Allen Institute for AI WildGuard — one-stop safety moderation judge.

    A generative classifier that evaluates, in a single pass: (1) whether the user
    request is harmful, (2) whether the assistant response is a refusal, and
    (3) whether the assistant response is harmful. ``valid`` is ``False`` when the
    request is harmful or (when an ``output_text`` is supplied) the response is harmful.
    The three signals are surfaced as ``categories``. Fails closed
    (``valid=False`` with ``extra={"parse_failure": True}``) when no field parses.

    For more information, see the
    [WildGuard model card](https://huggingface.co/allenai/wildguard).

    Args:
        model_id: Optional HuggingFace model ID. Defaults to ``allenai/wildguard``.
        provider: Optional pre-configured provider. Defaults to a ``HuggingFaceProvider``
            loading a causal LM.

    """

    SUPPORTED_MODELS: ClassVar = ["allenai/wildguard"]

    def __init__(self, model_id: str | None = None, provider: StandardProvider | None = None) -> None:
        """Initialize the WildGuard guardrail."""
        self.model_id = default(model_id, self.SUPPORTED_MODELS)
        load_kwargs: AnyDict = {}
        if provider is not None:
            self.provider = provider
            if isinstance(self.provider, HuggingFaceProvider):
                from transformers import AutoModelForCausalLM, AutoTokenizer

                load_kwargs = {"model_class": AutoModelForCausalLM, "tokenizer_class": AutoTokenizer}
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.provider = HuggingFaceProvider(model_class=AutoModelForCausalLM, tokenizer_class=AutoTokenizer)
        self.provider.load_model(self.model_id, **load_kwargs)

    def validate(  # type: ignore[override]
        self, input_text: str, output_text: str | None = None, **kwargs: Any
    ) -> GuardrailOutput:
        """Classify ``input_text`` (and optionally an assistant ``output_text``)."""
        result = super().validate(input_text, output_text=output_text, **kwargs)
        if isinstance(result, list):
            msg = "WildGuard.validate received a list input but only supports single strings."
            raise TypeError(msg)
        return result

    def _pre_processing(
        self, input_text: str, output_text: str | None = None, **kwargs: Any
    ) -> GuardrailPreprocessOutput[WildGuardPreprocessData]:
        del kwargs
        prompt = WILDGUARD_FORMAT.format(prompt=input_text, response=output_text or "")
        return GuardrailPreprocessOutput(
            data={"messages": [{"role": "user", "content": prompt}], "has_response": output_text is not None}
        )

    def _inference(
        self, model_inputs: GuardrailPreprocessOutput[WildGuardPreprocessData]
    ) -> GuardrailInferenceOutput[WildGuardInferenceData]:
        result = self.provider.generate_chat(
            messages=model_inputs.data["messages"],
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            apply_chat_template=False,
        )
        # Carry has_response through so _post_processing can fail closed on an
        # unparsed response verdict (rather than treating it as safe).
        result.data["has_response"] = model_inputs.data["has_response"]
        return result

    def _post_processing(self, model_outputs: GuardrailInferenceOutput[WildGuardInferenceData]) -> GuardrailOutput:
        text = model_outputs.data["generated_text"]
        has_response = model_outputs.data.get("has_response", False)
        harmful_request = _field(_HARMFUL_REQUEST, text)
        response_refusal = _field(_RESPONSE_REFUSAL, text)
        harmful_response = _field(_HARMFUL_RESPONSE, text)
        # Fail closed if the always-present request verdict is missing, or if a response was
        # being judged but its harm verdict didn't parse (don't silently pass it as safe).
        if harmful_request is None or (has_response and harmful_response is None):
            return GuardrailOutput(valid=False, explanation=text, extra={"parse_failure": True})
        categories = [
            CategoryResult(name="harmful_request", triggered=harmful_request),
            CategoryResult(name="harmful_response", triggered=harmful_response),
            CategoryResult(name="response_refusal", triggered=response_refusal),
        ]
        return GuardrailOutput(
            valid=not (bool(harmful_request) or bool(harmful_response)),
            explanation=text,
            categories=categories,
            usage=GuardrailUsage(
                prompt_tokens=model_outputs.data.get("prompt_token_count"),
                completion_tokens=model_outputs.data.get("completion_token_count"),
            ),
        )
