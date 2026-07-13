import re
from typing import Any, ClassVar

from any_guardrail.base import GuardrailName, GuardrailOutput, ThreeStageGuardrail
from any_guardrail.guardrails.utils import default
from any_guardrail.prompt_registry import PROMPT_REGISTRY, resolve_prompt
from any_guardrail.prompts import PromptSpec, PromptTemplate
from any_guardrail.providers.base import StandardProvider
from any_guardrail.providers.huggingface import HuggingFaceProvider
from any_guardrail.registry import GUARDRAIL_METADATA
from any_guardrail.taxonomy import GuardrailMetadata
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

POLYGUARD_SYSTEM_PROMPT = PROMPT_REGISTRY[GuardrailName.POLY_GUARD].resolve().segments["system"]
"""PolyGuard moderation system prompt (registry-sourced)."""
POLYGUARD_USER_TEMPLATE = PROMPT_REGISTRY[GuardrailName.POLY_GUARD].resolve().segments["user"]
"""PolyGuard user template (registry-sourced); fills ``{prompt}`` / ``{response}``."""

MAX_NEW_TOKENS = 100

_HARMFUL_REQUEST = re.compile(r"Harmful request:\s*(yes|no)", re.IGNORECASE)
_RESPONSE_REFUSAL = re.compile(r"Response refusal:\s*(yes|no)", re.IGNORECASE)
_HARMFUL_RESPONSE = re.compile(r"Harmful response:\s*(yes|no)", re.IGNORECASE)
_CODE_PATTERN = re.compile(r"S\d{1,2}")


def _field(pattern: re.Pattern[str], text: str) -> bool | None:
    match = pattern.search(text)
    return match.group(1).strip().lower() == "yes" if match else None


class PolyGuard(ThreeStageGuardrail[PolyGuardPreprocessData, PolyGuardInferenceData]):
    """Multilingual safety-moderation judge reporting request harm, response harm, and refusal across 17 languages.

    Generative classifier (fine-tuned Ministral / Qwen decoder LLMs) that, given a human request
    and an optional assistant response, reports three boolean signals — whether the request is
    harmful, whether the response is a refusal, and whether the response is harmful — plus the
    MLCommons hazard categories (``S1`` ... ``S14``) violated. Trained for moderation across 17
    languages.

    Verdict mapping onto ``GuardrailOutput``:

    - ``valid`` is ``False`` when the request or the response is judged harmful.
    - ``categories`` carries the three boolean signals (``harmful_request``, ``harmful_response``,
      ``response_refusal``) plus one ``CategoryResult`` per violated hazard code (``name`` = ``Sx``,
      ``description`` = the taxonomy label, ``triggered=True``), deduplicated in order of appearance.
    - ``explanation`` is the raw generation.
    - ``usage`` carries the prompt / completion token counts. No canonical ``score`` or ``spans``
      are produced.
    - Fails closed (``valid=False`` with ``extra={"parse_failure": True}``) when neither
      harmfulness field parses.

    Expected inputs: a single ``input_text`` string (the human request) plus an optional
    ``output_text`` string (the assistant response). The prompt template always carries both slots,
    so ``output_text`` defaults to an empty response when omitted; supply it to have the response
    judged for harm and refusal. Single strings only — passing a list raises ``TypeError``.

    For more information, see the model cards:

    - [ToxicityPrompts/PolyGuard-Ministral](https://huggingface.co/ToxicityPrompts/PolyGuard-Ministral) (default).
    - [ToxicityPrompts/PolyGuard-Qwen](https://huggingface.co/ToxicityPrompts/PolyGuard-Qwen).
    - [ToxicityPrompts/PolyGuard-Qwen-Smol](https://huggingface.co/ToxicityPrompts/PolyGuard-Qwen-Smol).

    Args:
        model_id: Optional HuggingFace model ID; must be one of ``SUPPORTED_MODELS``. Defaults to
            ``ToxicityPrompts/PolyGuard-Ministral``.
        provider: Optional pre-configured provider. Defaults to a ``HuggingFaceProvider`` loading a
            causal LM.

    """

    SUPPORTED_MODELS: ClassVar = [
        "ToxicityPrompts/PolyGuard-Ministral",
        "ToxicityPrompts/PolyGuard-Qwen",
        "ToxicityPrompts/PolyGuard-Qwen-Smol",
    ]

    METADATA: ClassVar[GuardrailMetadata] = GUARDRAIL_METADATA[GuardrailName.POLY_GUARD]

    PROMPT: ClassVar[PromptSpec] = PROMPT_REGISTRY[GuardrailName.POLY_GUARD]

    def __init__(
        self,
        model_id: str | None = None,
        provider: StandardProvider | None = None,
        prompt: PromptTemplate | None = None,
        prompt_version: str | None = None,
    ) -> None:
        """Initialize the PolyGuard guardrail.

        Args:
            model_id: Optional HuggingFace model ID; must be one of ``SUPPORTED_MODELS``. Defaults
                to ``ToxicityPrompts/PolyGuard-Ministral``; ``ToxicityPrompts/PolyGuard-Qwen`` and
                ``ToxicityPrompts/PolyGuard-Qwen-Smol`` are the Qwen-based alternatives.
            provider: Optional pre-configured provider. When ``None``, a ``HuggingFaceProvider`` is
                built targeting a causal LM (``AutoModelForCausalLM`` + ``AutoTokenizer``). A
                supplied ``HuggingFaceProvider`` is corrected to those classes at load time; any
                other provider is used as-is.
            prompt: Optional prompt-template override, used as-is (system prompt plus a user
                template filling ``{prompt}`` / ``{response}``). Defaults to ``None`` — the registry
                default, or the version named by ``prompt_version``.
            prompt_version: Registered prompt version to use when ``prompt`` is not given. Defaults
                to ``None`` (the default version). See ``AnyGuardrail.list_prompt_versions``.

        """
        self.model_id = default(model_id, self.SUPPORTED_MODELS)
        self._prompt = resolve_prompt(GuardrailName.POLY_GUARD, prompt, prompt_version)
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
        """Classify ``input_text`` and, optionally, an assistant ``output_text``.

        Args:
            input_text: The human request to moderate. Single string only.
            output_text: Optional assistant response, judged for harm and refusal alongside the
                request. When omitted, the response slot is left empty.
            **kwargs: Forwarded to the underlying three-stage pipeline; unused by this guardrail.

        Returns:
            GuardrailOutput with ``valid=False`` when the request or response is harmful,
            ``categories`` holding the ``harmful_request`` / ``harmful_response`` /
            ``response_refusal`` booleans and any violated hazard codes, and ``explanation`` set to
            the raw generation.

        Raises:
            TypeError: If a list input is supplied — only single strings are supported.

        """
        result = super().validate(input_text, output_text=output_text, **kwargs)
        if isinstance(result, list):
            msg = "PolyGuard.validate received a list input but only supports single strings."
            raise TypeError(msg)
        return result

    def _pre_processing(
        self, input_text: str, output_text: str | None = None, **kwargs: Any
    ) -> GuardrailPreprocessOutput[PolyGuardPreprocessData]:
        """Build the system + user chat messages for the classifier.

        Args:
            input_text: The human request, formatted into the ``Human user`` slot.
            output_text: Optional assistant response, formatted into the ``AI assistant`` slot;
                defaults to an empty string when omitted.
            **kwargs: Ignored (discarded via ``del kwargs``).

        Returns:
            GuardrailPreprocessOutput wrapping ``{"messages": ...}``.

        """
        del kwargs
        user = self._prompt.segments["user"].format(prompt=input_text, response=output_text or "")
        messages: ChatMessages = [
            {"role": "system", "content": self._prompt.segments["system"]},
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
