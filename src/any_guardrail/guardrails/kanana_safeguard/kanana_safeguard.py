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

KananaPreprocessData = AnyDict
KananaInferenceData = AnyDict

# Per-model unsafe-token taxonomies. Each model emits a single token: ``<SAFE>`` or an
# ``<UNSAFE-*>`` code. The harm model also evaluates the assistant turn when supplied.
KANANA_CATEGORIES: dict[str, dict[str, str]] = {
    "kakaocorp/kanana-safeguard-8b": {
        "S1": "Hate",
        "S2": "Harassment",
        "S3": "Sexual Content",
        "S4": "Crime",
        "S5": "Child Sexual Abuse",
        "S6": "Self-Harm",
        "S7": "Misinformation",
    },
    "kakaocorp/kanana-safeguard-siren-8b": {
        "I1": "Adult Authentication",
        "I2": "Professional Advice",
        "I3": "Personal Information",
        "I4": "Intellectual Property",
    },
    "kakaocorp/kanana-safeguard-prompt-2.1b": {
        "A1": "Prompt Injection",
        "A2": "Prompt Leaking",
    },
}
# The harm model is the only variant trained to judge an assistant turn.
_EVALUATES_RESPONSE = frozenset({"kakaocorp/kanana-safeguard-8b"})

MAX_NEW_TOKENS = 4
_TOKEN_PATTERN = re.compile(r"<(SAFE|UNSAFE-([A-Z]\d+))>")


class KananaSafeguard(ThreeStageGuardrail[KananaPreprocessData, KananaInferenceData]):
    """Kakao Kanana Safeguard — Korean safety guardrails.

    Decoder LLMs that emit a single verdict token: ``<SAFE>`` or an ``<UNSAFE-*>`` code.
    Three variants cover different taxonomies: harmful content (``-8b``, also judges an
    assistant turn), legal/policy risk (``-siren-8b``), and prompt attacks (``-prompt-2.1b``).
    ``valid`` is ``True`` on ``<SAFE>``; the matched code is surfaced in ``categories``.
    Fails closed (``valid=False`` with ``extra={"parse_failure": True}``) when no token parses.

    For more information, see the model cards:

    - [kanana-safeguard-8b](https://huggingface.co/kakaocorp/kanana-safeguard-8b) (default).
    - [kanana-safeguard-siren-8b](https://huggingface.co/kakaocorp/kanana-safeguard-siren-8b).
    - [kanana-safeguard-prompt-2.1b](https://huggingface.co/kakaocorp/kanana-safeguard-prompt-2.1b).

    Args:
        model_id: Optional HuggingFace model ID. Defaults to ``kakaocorp/kanana-safeguard-8b``.
        provider: Optional pre-configured provider. Defaults to a ``HuggingFaceProvider``
            loading a causal LM.
    """

    SUPPORTED_MODELS: ClassVar = [
        "kakaocorp/kanana-safeguard-8b",
        "kakaocorp/kanana-safeguard-siren-8b",
        "kakaocorp/kanana-safeguard-prompt-2.1b",
    ]

    def __init__(self, model_id: str | None = None, provider: StandardProvider | None = None) -> None:
        """Initialize the Kanana Safeguard guardrail."""
        self.model_id = default(model_id, self.SUPPORTED_MODELS)
        self.categories = KANANA_CATEGORIES[self.model_id]
        self.evaluates_response = self.model_id in _EVALUATES_RESPONSE
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
        """Classify ``input_text`` (and, for the harm model, an assistant ``output_text``)."""
        result = super().validate(input_text, output_text=output_text, **kwargs)
        if isinstance(result, list):
            msg = "KananaSafeguard.validate received a list input but only supports single strings."
            raise TypeError(msg)
        return result

    def _pre_processing(
        self, input_text: str, output_text: str | None = None, **kwargs: Any
    ) -> GuardrailPreprocessOutput[KananaPreprocessData]:
        del kwargs
        messages: ChatMessages = [{"role": "user", "content": input_text}]
        if output_text is not None and self.evaluates_response:
            messages.append({"role": "assistant", "content": output_text})
        return GuardrailPreprocessOutput(data={"messages": messages})

    def _inference(
        self, model_inputs: GuardrailPreprocessOutput[KananaPreprocessData]
    ) -> GuardrailInferenceOutput[KananaInferenceData]:
        # Keep special tokens: the verdict itself (``<SAFE>``/``<UNSAFE-*>``) is a special token.
        return self.provider.generate_chat(
            messages=model_inputs.data["messages"],
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            skip_special_tokens=False,
        )

    def _post_processing(
        self, model_outputs: GuardrailInferenceOutput[KananaInferenceData]
    ) -> GuardrailOutput:
        text = model_outputs.data["generated_text"]
        match = _TOKEN_PATTERN.search(text)
        if match is None:
            return GuardrailOutput(valid=False, explanation=text, extra={"parse_failure": True})
        usage = GuardrailUsage(
            prompt_tokens=model_outputs.data.get("prompt_token_count"),
            completion_tokens=model_outputs.data.get("completion_token_count"),
        )
        if match.group(1) == "SAFE":
            return GuardrailOutput(valid=True, explanation=text, usage=usage)
        code = match.group(2)
        return GuardrailOutput(
            valid=False,
            explanation=text,
            categories=[CategoryResult(name=code, description=self.categories.get(code), triggered=True)],
            extra={"verdict": match.group(0)},
            usage=usage,
        )
