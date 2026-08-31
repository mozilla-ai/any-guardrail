import json
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

# NVIDIA content-safety taxonomy as shipped with Llama-3.1-Nemotron-Safety-Guard-8B-v3
# (23 categories). The numbering is NOT the same as NEMOTRON_CATEGORIES above: this model
# inserts "Other" at S14, shifting Manipulation..Immoral/Unethical down by one.
NEMOTRON_8B_V3_CATEGORIES = {
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
    "S14": "Other",
    "S15": "Manipulation",
    "S16": "Fraud/Deception",
    "S17": "Malware",
    "S18": "High Risk Gov Decision Making",
    "S19": "Political/Misinformation/Conspiracy",
    "S20": "Copyright/Trademark/Plagiarism",
    "S21": "Unauthorized Advice",
    "S22": "Illegal Activity",
    "S23": "Immoral/Unethical",
}

# Every category line ends with a trailing space in NVIDIA's published template; it is produced
# here rather than written literally so the trailing-whitespace hook can't strip it.
_8B_V3_TAXONOMY = "\n".join(f"{code}: {name}. " for code, name in NEMOTRON_8B_V3_CATEGORIES.items())
NEMOTRON_8B_V3_INSTRUCTION = (
    "Task: Check if there is unsafe content in the user or agent messages in the below "
    "conversation in accordance with our safety policy with the below categories. Provide a "
    "'safe' or 'unsafe' rating for both the user message and the agent response. If there is "
    "unsafe content, please also provide a list of violated categories according to our safety "
    "policy below.\n\n"
    f"<BEGIN UNSAFE CONTENT CATEGORIES>\n{_8B_V3_TAXONOMY}\n<END UNSAFE CONTENT CATEGORIES>"
)
NEMOTRON_8B_V3_OUTPUT_FORMAT = (
    "Provide your safety assessment for the above conversation in the below output JSON format:\n"
    '{"User Safety": safe or unsafe rating for the user message, "Response Safety": safe or '
    "unsafe rating for the agent response. Omit if no agent response present. "
    '"Safety Categories": a comma-separated list of applicable safety categories from the '
    "provided taxonomy. Omit if all safe.}\n"
    "\n"
    "Do not include anything other than the output JSON in your response.\n"
    "Output JSON: "
)

MODEL_8B_V3 = "nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3"

MAX_NEW_TOKENS = 400
# The 8B-v3 emits only a small JSON object and has no reasoning mode, so it needs far less room.
MAX_NEW_TOKENS_8B_V3 = 100

_PROMPT_HARM = re.compile(r"Prompt harm:\s*(harmful|unharmful)", re.IGNORECASE)
_RESPONSE_HARM = re.compile(r"Response Harm:\s*(harmful|unharmful)", re.IGNORECASE)
_THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)
# The model is told to emit nothing but the JSON object, but llamafile/HF generations sometimes
# wrap it in a ```json fence, so pull out the first balanced-looking {...} span rather than
# json.loads()-ing the whole generation.
_JSON_OBJECT = re.compile(r"\{.*?\}", re.DOTALL)


def _field(pattern: re.Pattern[str], text: str) -> bool | None:
    match = pattern.search(text)
    return match.group(1).strip().lower() == "harmful" if match else None


def _safety_verdict(value: object) -> bool | None:
    """Map an 8B-v3 ``"safe"``/``"unsafe"`` rating onto the harmful boolean the output uses."""
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if normalized == "unsafe":
        return True
    return False if normalized == "safe" else None


class NemotronContentSafety(ThreeStageGuardrail[NemotronPreprocessData, NemotronInferenceData]):
    """Safety classifier covering NVIDIA's multi-category content-safety taxonomy.

    Decoder LLM that classifies a user prompt and an optional assistant response against NVIDIA's
    content-safety taxonomy. Two variants are supported, and they differ in base model, taxonomy,
    and output format:

    - ``nvidia/Nemotron-Content-Safety-Reasoning-4B`` (default, Gemma-3-4B base) — 22 categories
      (``S1`` Violence ... ``S22`` Immoral/Unethical). Prompted to emit
      ``Prompt harm: harmful/unharmful`` and ``Response Harm: harmful/unharmful``; with
      ``think=True`` it first reasons inside ``<think>...</think>`` (stripped before parsing).
    - ``nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3`` (Llama-3.1-8B base) — NVIDIA's own
      published prompt over a 23-category taxonomy (``Other`` is inserted at ``S14``, shifting
      the rest down one), answering with a JSON object holding ``"User Safety"``,
      ``"Response Safety"``, and ``"Safety Categories"``. Multilingual (9 trained languages, ~20
      zero-shot). It has no reasoning mode, so ``think=True`` is rejected.

    Verdict mapping onto ``GuardrailOutput`` (identical for both variants):

    - ``valid`` is ``False`` when either the prompt or the response is judged harmful.
    - ``categories`` carries two boolean signals — ``prompt_harm`` and ``response_harm``
      (``triggered`` reflects each verdict).
    - ``explanation`` is the raw generation (including any ``<think>`` reasoning).
    - ``usage`` carries the prompt / completion token counts. No canonical ``score`` or ``spans``
      are produced.
    - Fails closed (``valid=False`` with ``extra={"parse_failure": True}``) when the prompt verdict
      is missing, or when a response was judged but its verdict did not parse.
    - On the 8B-v3 variant only, ``extra["safety_categories"]`` additionally lists the violated
      taxonomy category names the model reported.

    Expected inputs: a single ``input_text`` prompt string plus an optional ``output_text``
    assistant response; when ``output_text`` is given the response is moderated alongside the
    prompt. Single strings only — passing a list raises ``TypeError``.

    The 4B variant is distributed under the NVIDIA Open Model License and the Gemma Terms of Use;
    the 8B-v3 variant under the NVIDIA Open Model License and the Llama 3.1 Community License.

    For more information, see the
    [nvidia/Nemotron-Content-Safety-Reasoning-4B model card](https://huggingface.co/nvidia/Nemotron-Content-Safety-Reasoning-4B)
    and the
    [nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3 model card](https://huggingface.co/nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3).

    Args:
        think: If ``True``, request chain-of-thought reasoning (appends ``/think``); otherwise
            ``/no_think``. Reasoning is stripped from the verdict but retained in ``explanation``.
            Only the 4B variant supports it.
        model_id: Optional HuggingFace model ID; must be one of ``SUPPORTED_MODELS``. Defaults to
            ``nvidia/Nemotron-Content-Safety-Reasoning-4B``.
        provider: Optional pre-configured provider. Defaults to a ``HuggingFaceProvider`` loading a
            causal LM; pass a ``LlamafileProvider`` to run a GGUF build of the 8B-v3 variant
            instead (it is the variant with a published llamafile artifact).

    """

    SUPPORTED_MODELS: ClassVar = [
        "nvidia/Nemotron-Content-Safety-Reasoning-4B",
        MODEL_8B_V3,
    ]

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
                is stripped before parsing but kept in ``GuardrailOutput.explanation``. Supported
                only by the 4B variant.
            model_id: Optional HuggingFace model ID; must be one of ``SUPPORTED_MODELS``. Defaults
                to ``nvidia/Nemotron-Content-Safety-Reasoning-4B``.
            provider: Optional pre-configured provider. When ``None``, a ``HuggingFaceProvider`` is
                built targeting a causal LM (``AutoModelForCausalLM`` + ``AutoTokenizer``). A
                supplied ``HuggingFaceProvider`` is corrected to those classes at load time; any
                other provider is used as-is.

        Raises:
            ValueError: If ``model_id`` is not in ``SUPPORTED_MODELS``, or if ``think=True`` is
                combined with the 8B-v3 variant, which has no reasoning mode.

        """
        self.model_id = default(model_id, self.SUPPORTED_MODELS)
        if think and self.model_id == MODEL_8B_V3:
            msg = (
                f"think=True is not supported by {MODEL_8B_V3}: it has no reasoning mode and "
                f"answers with a JSON verdict only. Use "
                f"nvidia/Nemotron-Content-Safety-Reasoning-4B for chain-of-thought reasoning."
            )
            raise ValueError(msg)
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
        if self.model_id == MODEL_8B_V3:
            body = self._build_8b_v3_prompt(input_text, output_text)
        else:
            directive = "/think" if self.think else "/no_think"
            body = f"{NEMOTRON_INSTRUCTION}\n\nUser prompt:\n{input_text}"
            if output_text is not None:
                body += f"\n\nAI response:\n{output_text}"
            body += f"\n\n{directive}"
        messages: ChatMessages = [{"role": "user", "content": body}]
        return GuardrailPreprocessOutput(data={"messages": messages, "has_response": output_text is not None})

    @staticmethod
    def _build_8b_v3_prompt(input_text: str, output_text: str | None) -> str:
        """Reproduce NVIDIA's published 8B-v3 jinja template byte-for-byte.

        The blank lines around the conversation turns are not cosmetic — they are what the
        ``{% if response %}`` / ``{% endif %}`` block tags leave behind when the upstream template
        renders, so they are part of the prompt the model was trained on.
        """
        conversation = f"<BEGIN CONVERSATION>\n\nuser: {input_text}\n\n"
        if output_text is not None:
            conversation += f"response: agent: {output_text}\n\n"
        conversation += "<END CONVERSATION>"
        return f"{NEMOTRON_8B_V3_INSTRUCTION}\n\n{conversation}\n\n{NEMOTRON_8B_V3_OUTPUT_FORMAT}"

    def _inference(
        self, model_inputs: GuardrailPreprocessOutput[NemotronInferenceData]
    ) -> GuardrailInferenceOutput[NemotronInferenceData]:
        max_new_tokens = MAX_NEW_TOKENS_8B_V3 if self.model_id == MODEL_8B_V3 else MAX_NEW_TOKENS
        result = self.provider.generate_chat(
            messages=model_inputs.data["messages"], max_new_tokens=max_new_tokens, do_sample=False
        )
        # Carry has_response through so _post_processing can fail closed on an unparsed response verdict.
        result.data["has_response"] = model_inputs.data["has_response"]
        return result

    def _post_processing(self, model_outputs: GuardrailInferenceOutput[NemotronInferenceData]) -> GuardrailOutput:
        text = model_outputs.data["generated_text"]
        has_response = model_outputs.data.get("has_response", False)
        if self.model_id == MODEL_8B_V3:
            prompt_harm, response_harm, safety_categories = self._parse_8b_v3(text)
        else:
            without_think = _THINK_PATTERN.sub("", text).strip()
            prompt_harm = _field(_PROMPT_HARM, without_think)
            response_harm = _field(_RESPONSE_HARM, without_think)
            safety_categories = None
        # Fail closed if the prompt verdict is missing, or a judged response's verdict didn't parse.
        if prompt_harm is None or (has_response and response_harm is None):
            return GuardrailOutput(valid=False, explanation=text, extra={"parse_failure": True})
        extra: AnyDict | None = None if safety_categories is None else {"safety_categories": safety_categories}
        return GuardrailOutput(
            valid=not (bool(prompt_harm) or bool(response_harm)),
            explanation=text,
            categories=[
                CategoryResult(name="prompt_harm", triggered=prompt_harm),
                CategoryResult(name="response_harm", triggered=response_harm),
            ],
            extra=extra,
            usage=GuardrailUsage(
                prompt_tokens=model_outputs.data.get("prompt_token_count"),
                completion_tokens=model_outputs.data.get("completion_token_count"),
            ),
        )

    @staticmethod
    def _parse_8b_v3(text: str) -> tuple[bool | None, bool | None, list[str]]:
        """Pull the prompt/response verdicts and violated categories out of the 8B-v3 JSON reply.

        Returns ``(prompt_harm, response_harm, safety_categories)``. A ``None`` verdict means the
        field was missing or unrecognized, which the caller turns into a fail-closed result;
        ``"Response Safety"`` is legitimately absent when no response was moderated.
        """
        match = _JSON_OBJECT.search(text)
        if match is None:
            return None, None, []
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None, None, []
        if not isinstance(payload, dict):
            return None, None, []
        raw_categories = payload.get("Safety Categories")
        categories = (
            [item.strip() for item in raw_categories.split(",") if item.strip()]
            if isinstance(raw_categories, str)
            else []
        )
        return _safety_verdict(payload.get("User Safety")), _safety_verdict(payload.get("Response Safety")), categories
