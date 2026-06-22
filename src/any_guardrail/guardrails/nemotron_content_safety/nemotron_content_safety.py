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
    """NVIDIA Nemotron Content Safety Reasoning — 4B safety classifier with optional reasoning.

    Decoder LLM (Gemma-3-4B base) that classifies a prompt and optional response against
    NVIDIA's 22-category content-safety taxonomy. ``valid`` is ``False`` when the prompt or
    response is harmful. With ``think=True`` the model reasons inside ``<think>...</think>``
    before the verdict (stripped before parsing). Fails closed (``valid=False`` with
    ``extra={"parse_failure": True}``) when no verdict parses. Distributed under the
    NVIDIA Open Model License + Gemma Terms.

    For more information, see the
    [Nemotron-Content-Safety-Reasoning-4B model card](https://huggingface.co/nvidia/Nemotron-Content-Safety-Reasoning-4B).

    Args:
        think: If ``True``, request chain-of-thought reasoning (``/think``); otherwise ``/no_think``.
        model_id: Optional HuggingFace model ID. Defaults to ``nvidia/Nemotron-Content-Safety-Reasoning-4B``.
        provider: Optional pre-configured provider. Defaults to a ``HuggingFaceProvider``
            loading a causal LM.

    """

    SUPPORTED_MODELS: ClassVar = ["nvidia/Nemotron-Content-Safety-Reasoning-4B"]

    def __init__(
        self,
        think: bool = False,
        model_id: str | None = None,
        provider: StandardProvider | None = None,
    ) -> None:
        """Initialize the Nemotron Content Safety guardrail."""
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
        """Classify ``input_text`` (and optionally an assistant ``output_text``)."""
        result = super().validate(input_text, output_text=output_text, **kwargs)
        if isinstance(result, list):
            msg = "NemotronContentSafety.validate received a list input but only supports single strings."
            raise TypeError(msg)
        return result

    def _pre_processing(
        self, input_text: str, output_text: str | None = None, **kwargs: Any
    ) -> GuardrailPreprocessOutput[NemotronPreprocessData]:
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
