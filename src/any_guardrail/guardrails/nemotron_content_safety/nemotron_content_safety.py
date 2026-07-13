import re
from typing import Any, ClassVar

from any_guardrail.base import GuardrailName, GuardrailOutput, ThreeStageGuardrail
from any_guardrail.guardrails.utils import default
from any_guardrail.prompt_registry import PROMPT_REGISTRY
from any_guardrail.prompts import PromptSpec
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

NemotronPreprocessData = AnyDict
NemotronInferenceData = AnyDict

# NVIDIA content-safety taxonomy (22 categories).
NEMOTRON_CATEGORIES = {
    "S1": "Violence",
    "S2": "Sexual",
    "S3": "Criminal Planning/Confessions",
    "S4": "Guns and Illegal Weapons",
    "S5": "Controlled/Regulated Substances",
    "S6": "Suicide and Self Harm",
    "S7": "Sexual (minor)",
    "S8": "Hate/Identity Hate",
    "S9": "PII/Privacy",
    "S10": "Harassment",
    "S11": "Threat",
    "S12": "Profanity",
    "S13": "Needs Caution",
    "S14": "Manipulation",
    "S15": "Fraud/Deception",
    "S16": "Malware",
    "S17": "High Risk Gov Decision Making",
    "S18": "Political/Misinformation/Conspiracy",
    "S19": "Copyright/Trademark/Plagiarism",
    "S20": "Unauthorized Advice",
    "S21": "Illegal Activity",
    "S22": "Immoral/Unethical",
}

_TAXONOMY = "\n".join(f"{code}: {name}" for code, name in NEMOTRON_CATEGORIES.items())
NEMOTRON_INSTRUCTION = (
    "You are a content safety classifier. Using the safety categories below, decide whether the "
    "user prompt and (if present) the AI response are harmful.\n\n"
    f"Safety categories:\n{_TAXONOMY}\n\n"
    "Respond with exactly:\nPrompt harm: harmful/unharmful\nResponse Harm: harmful/unharmful"
)

MAX_NEW_TOKENS = 400

_PROMPT_HARM = re.compile(r"Prompt harm:\s*(harmful|unharmful)", re.IGNORECASE)
_RESPONSE_HARM = re.compile(r"Response Harm:\s*(harmful|unharmful)", re.IGNORECASE)
_THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)


def _field(pattern: re.Pattern[str], text: str) -> bool | None:
    match = pattern.search(text)
    return match.group(1).strip().lower() == "harmful" if match else None


class NemotronContentSafety(ThreeStageGuardrail[NemotronPreprocessData, NemotronInferenceData]):
    """Reasoning safety classifier covering a 22-category content-safety taxonomy.

    Decoder LLM (Gemma-3-4B base) that classifies a user prompt and an optional assistant response
    against NVIDIA's 22-category content-safety taxonomy (``S1`` Violence ... ``S22``
    Immoral/Unethical). The model is prompted to emit ``Prompt harm: harmful/unharmful`` and
    ``Response Harm: harmful/unharmful``; with ``think=True`` it first reasons inside
    ``<think>...</think>`` (stripped before the verdict is parsed).

    Verdict mapping onto ``GuardrailOutput``:

    - ``valid`` is ``False`` when either the prompt or the response is judged harmful.
    - ``categories`` carries two boolean signals — ``prompt_harm`` and ``response_harm``
      (``triggered`` reflects each verdict).
    - ``explanation`` is the raw generation (including any ``<think>`` reasoning).
    - ``usage`` carries the prompt / completion token counts. No canonical ``score`` or ``spans``
      are produced.
    - Fails closed (``valid=False`` with ``extra={"parse_failure": True}``) when the prompt verdict
      is missing, or when a response was judged but its verdict did not parse.

    Expected inputs: a single ``input_text`` prompt string plus an optional ``output_text``
    assistant response; when ``output_text`` is given the response is moderated alongside the
    prompt. Single strings only — passing a list raises ``TypeError``.

    Distributed under the NVIDIA Open Model License and the Gemma Terms of Use.

    For more information, see the
    [nvidia/Nemotron-Content-Safety-Reasoning-4B model card](https://huggingface.co/nvidia/Nemotron-Content-Safety-Reasoning-4B).

    Args:
        think: If ``True``, request chain-of-thought reasoning (appends ``/think``); otherwise
            ``/no_think``. Reasoning is stripped from the verdict but retained in ``explanation``.
        model_id: Optional HuggingFace model ID; must be one of ``SUPPORTED_MODELS``. Defaults to
            ``nvidia/Nemotron-Content-Safety-Reasoning-4B``.
        provider: Optional pre-configured provider. Defaults to a ``HuggingFaceProvider`` loading a
            causal LM.

    """

    SUPPORTED_MODELS: ClassVar = ["nvidia/Nemotron-Content-Safety-Reasoning-4B"]

    METADATA: ClassVar[GuardrailMetadata] = GUARDRAIL_METADATA[GuardrailName.NEMOTRON_CONTENT_SAFETY]

    # Reference-only: the instruction is assembled at runtime (see NEMOTRON_INSTRUCTION); the
    # registry entry is for discovery/pinning and is not user-overridable.
    PROMPT: ClassVar[PromptSpec] = PROMPT_REGISTRY[GuardrailName.NEMOTRON_CONTENT_SAFETY]

    def __init__(
        self,
        think: bool = False,
        model_id: str | None = None,
        provider: StandardProvider | None = None,
    ) -> None:
        """Initialize the Nemotron Content Safety guardrail.

        Args:
            think: If ``True``, request chain-of-thought reasoning (``/think``) before the verdict;
                otherwise ``/no_think``. Slower but can improve borderline judgments; the reasoning
                is stripped before parsing but kept in ``GuardrailOutput.explanation``.
            model_id: Optional HuggingFace model ID; must be one of ``SUPPORTED_MODELS``. Defaults
                to ``nvidia/Nemotron-Content-Safety-Reasoning-4B``.
            provider: Optional pre-configured provider. When ``None``, a ``HuggingFaceProvider`` is
                built targeting a causal LM (``AutoModelForCausalLM`` + ``AutoTokenizer``). A
                supplied ``HuggingFaceProvider`` is corrected to those classes at load time; any
                other provider is used as-is.

        """
        self.model_id = default(model_id, self.SUPPORTED_MODELS)
        self.think = think
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
            input_text: The user prompt to moderate. Single string only.
            output_text: Optional assistant response moderated alongside the prompt. When provided,
                a missing or unparsable response verdict causes the guardrail to fail closed.
            **kwargs: Forwarded to the underlying three-stage pipeline; unused by this guardrail.

        Returns:
            GuardrailOutput with ``valid=False`` when the prompt or response is harmful,
            ``categories`` holding the ``prompt_harm`` / ``response_harm`` booleans, and
            ``explanation`` set to the raw generation.

        Raises:
            TypeError: If a list input is supplied — only single strings are supported.

        """
        result = super().validate(input_text, output_text=output_text, **kwargs)
        if isinstance(result, list):
            msg = "NemotronContentSafety.validate received a list input but only supports single strings."
            raise TypeError(msg)
        return result

    def _pre_processing(
        self, input_text: str, output_text: str | None = None, **kwargs: Any
    ) -> GuardrailPreprocessOutput[NemotronPreprocessData]:
        """Build the single-turn moderation chat message for the model.

        Args:
            input_text: The user prompt, embedded after the taxonomy instruction.
            output_text: Optional assistant response; when provided it is embedded as the
                ``AI response`` and a response verdict is expected.
            **kwargs: Ignored (discarded via ``del kwargs``).

        Returns:
            GuardrailPreprocessOutput wrapping ``{"messages": ..., "has_response": bool}``; the
            ``has_response`` flag lets ``_post_processing`` fail closed on an unparsed response
            verdict.

        """
        del kwargs
        directive = "/think" if self.think else "/no_think"
        body = f"{NEMOTRON_INSTRUCTION}\n\nUser prompt:\n{input_text}"
        if output_text is not None:
            body += f"\n\nAI response:\n{output_text}"
        body += f"\n\n{directive}"
        messages: ChatMessages = [{"role": "user", "content": body}]
        return GuardrailPreprocessOutput(data={"messages": messages, "has_response": output_text is not None})

    def _inference(
        self, model_inputs: GuardrailPreprocessOutput[NemotronInferenceData]
    ) -> GuardrailInferenceOutput[NemotronInferenceData]:
        result = self.provider.generate_chat(
            messages=model_inputs.data["messages"], max_new_tokens=MAX_NEW_TOKENS, do_sample=False
        )
        # Carry has_response through so _post_processing can fail closed on an unparsed response verdict.
        result.data["has_response"] = model_inputs.data["has_response"]
        return result

    def _post_processing(self, model_outputs: GuardrailInferenceOutput[NemotronInferenceData]) -> GuardrailOutput:
        text = model_outputs.data["generated_text"]
        has_response = model_outputs.data.get("has_response", False)
        without_think = _THINK_PATTERN.sub("", text).strip()
        prompt_harm = _field(_PROMPT_HARM, without_think)
        response_harm = _field(_RESPONSE_HARM, without_think)
        # Fail closed if the prompt verdict is missing, or a judged response's verdict didn't parse.
        if prompt_harm is None or (has_response and response_harm is None):
            return GuardrailOutput(valid=False, explanation=text, extra={"parse_failure": True})
        return GuardrailOutput(
            valid=not (bool(prompt_harm) or bool(response_harm)),
            explanation=text,
            categories=[
                CategoryResult(name="prompt_harm", triggered=prompt_harm),
                CategoryResult(name="response_harm", triggered=response_harm),
            ],
            usage=GuardrailUsage(
                prompt_tokens=model_outputs.data.get("prompt_token_count"),
                completion_tokens=model_outputs.data.get("completion_token_count"),
            ),
        )
