import re
from typing import Any, ClassVar

from any_guardrail.base import GuardrailOutput, ThreeStageGuardrail
from any_guardrail.guardrails.utils import default
from any_guardrail.providers.base import StandardProvider
from any_guardrail.providers.huggingface import HuggingFaceProvider
from any_guardrail.types import (
    AnyDict,
    CategoryResult,
    ChatMessages,
    GuardrailInferenceOutput,
    GuardrailPreprocessOutput,
    GuardrailUsage,
)

PolyGuardPreprocessData = AnyDict
PolyGuardInferenceData = AnyDict

# MLCommons hazard taxonomy used by PolyGuard's violation fields.
POLYGUARD_CATEGORIES = {
    "S1": "Violent Crimes",
    "S2": "Non-Violent Crimes",
    "S3": "Sex Crimes",
    "S4": "Child Exploitation",
    "S5": "Defamation",
    "S6": "Specialized Advice",
    "S7": "Privacy",
    "S8": "Intellectual Property",
    "S9": "Indiscriminate Weapons",
    "S10": "Hate",
    "S11": "Self-Harm",
    "S12": "Sexual Content",
    "S13": "Elections",
    "S14": "Code Interpreter Abuse",
}

POLYGUARD_SYSTEM_PROMPT = (
    "You are a classifier for analyzing interactions between humans and LLMs. Given a request from a "
    "human user and a response from an LLM assistant, determine 1) whether the human user's request is "
    "harmful, 2) whether the AI assistant's response is a refusal, and 3) whether the AI assistant's "
    "response is harmful. In addition, identify the policies violated if the interaction is unsafe and "
    "harmful according to the safety policy with the below categories."
)
POLYGUARD_USER_TEMPLATE = "Human user:\n{prompt}\n\nAI assistant:\n{response}"

MAX_NEW_TOKENS = 100

_HARMFUL_REQUEST = re.compile(r"Harmful request:\s*(yes|no)", re.IGNORECASE)
_RESPONSE_REFUSAL = re.compile(r"Response refusal:\s*(yes|no)", re.IGNORECASE)
_HARMFUL_RESPONSE = re.compile(r"Harmful response:\s*(yes|no)", re.IGNORECASE)
_CODE_PATTERN = re.compile(r"S\d{1,2}")


def _field(pattern: re.Pattern[str], text: str) -> bool | None:
    match = pattern.search(text)
    return match.group(1).strip().lower() == "yes" if match else None


class PolyGuard(ThreeStageGuardrail[PolyGuardPreprocessData, PolyGuardInferenceData]):
    """PolyGuard — multilingual safety moderation judge (17 languages).

    Generative classifier reporting request harmfulness, response refusal, response
    harmfulness, and the MLCommons policy categories violated. ``valid`` is ``False``
    when the request or response is harmful; violated S-codes and the boolean signals
    are surfaced as ``categories``. Fails closed (``valid=False`` with
    ``extra={"parse_failure": True}``) when no harmfulness field parses.

    For more information, see the model cards:

    - [PolyGuard-Ministral](https://huggingface.co/ToxicityPrompts/PolyGuard-Ministral) (default).
    - [PolyGuard-Qwen](https://huggingface.co/ToxicityPrompts/PolyGuard-Qwen).
    - [PolyGuard-Qwen-Smol](https://huggingface.co/ToxicityPrompts/PolyGuard-Qwen-Smol).

    Args:
        model_id: Optional HuggingFace model ID. Defaults to ``ToxicityPrompts/PolyGuard-Ministral``.
        provider: Optional pre-configured provider. Defaults to a ``HuggingFaceProvider``
            loading a causal LM.

    """

    SUPPORTED_MODELS: ClassVar = [
        "ToxicityPrompts/PolyGuard-Ministral",
        "ToxicityPrompts/PolyGuard-Qwen",
        "ToxicityPrompts/PolyGuard-Qwen-Smol",
    ]

    def __init__(self, model_id: str | None = None, provider: StandardProvider | None = None) -> None:
        """Initialize the PolyGuard guardrail."""
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
            msg = "PolyGuard.validate received a list input but only supports single strings."
            raise TypeError(msg)
        return result

    def _pre_processing(
        self, input_text: str, output_text: str | None = None, **kwargs: Any
    ) -> GuardrailPreprocessOutput[PolyGuardPreprocessData]:
        del kwargs
        user = POLYGUARD_USER_TEMPLATE.format(prompt=input_text, response=output_text or "")
        messages: ChatMessages = [
            {"role": "system", "content": POLYGUARD_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
        return GuardrailPreprocessOutput(data={"messages": messages})

    def _inference(
        self, model_inputs: GuardrailPreprocessOutput[PolyGuardPreprocessData]
    ) -> GuardrailInferenceOutput[PolyGuardInferenceData]:
        return self.provider.generate_chat(
            messages=model_inputs.data["messages"], max_new_tokens=MAX_NEW_TOKENS, do_sample=False
        )

    def _post_processing(self, model_outputs: GuardrailInferenceOutput[PolyGuardInferenceData]) -> GuardrailOutput:
        text = model_outputs.data["generated_text"]
        harmful_request = _field(_HARMFUL_REQUEST, text)
        response_refusal = _field(_RESPONSE_REFUSAL, text)
        harmful_response = _field(_HARMFUL_RESPONSE, text)
        if harmful_request is None and harmful_response is None:
            return GuardrailOutput(valid=False, explanation=text, extra={"parse_failure": True})
        categories = [
            CategoryResult(name="harmful_request", triggered=harmful_request),
            CategoryResult(name="harmful_response", triggered=harmful_response),
            CategoryResult(name="response_refusal", triggered=response_refusal),
        ]
        for code in dict.fromkeys(_CODE_PATTERN.findall(text)):
            if code in POLYGUARD_CATEGORIES:
                categories.append(CategoryResult(name=code, description=POLYGUARD_CATEGORIES[code], triggered=True))
        return GuardrailOutput(
            valid=not (bool(harmful_request) or bool(harmful_response)),
            explanation=text,
            categories=categories,
            usage=GuardrailUsage(
                prompt_tokens=model_outputs.data.get("prompt_token_count"),
                completion_tokens=model_outputs.data.get("completion_token_count"),
            ),
        )
