import re
from typing import Any, ClassVar

from any_guardrail.base import GuardrailName, GuardrailOutput, ThreeStageGuardrail
from any_guardrail.guardrails.utils import default
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
    """Korean safety decoder models covering harmful content, legal risk, and prompt attacks.

    Decoder LLMs, trained primarily for Korean text, that emit a single verdict token:
    ``<SAFE>`` or an ``<UNSAFE-*>`` code. Three variants cover different taxonomies:

    - ``kakaocorp/kanana-safeguard-8b`` (default): harmful content — Hate, Harassment,
      Sexual Content, Crime, Child Sexual Abuse, Self-Harm, Misinformation (``S1``-``S7``).
      The only variant trained to also judge an assistant turn (``output_text``).
    - ``kakaocorp/kanana-safeguard-siren-8b``: legal/policy risk — Adult Authentication,
      Professional Advice, Personal Information, Intellectual Property (``I1``-``I4``).
    - ``kakaocorp/kanana-safeguard-prompt-2.1b``: prompt attacks — Prompt Injection,
      Prompt Leaking (``A1``-``A2``).

    Inputs are single strings (no batching): ``validate(input_text)`` for the user turn,
    plus an optional assistant ``output_text`` that is only used by the harm (``-8b``)
    model; the other variants ignore it.

    ``GuardrailOutput`` mapping: ``valid`` is ``True`` on ``<SAFE>``. On an ``<UNSAFE-*>``
    verdict, ``categories`` holds one triggered entry named after the matched code (e.g.
    ``S3``) with its human-readable description, and ``extra["verdict"]`` carries the raw
    token. ``score`` is not populated — the single-token verdict has no probability.
    ``explanation`` is the raw generated text. Fails closed (``valid=False`` with
    ``extra={"parse_failure": True}``) when no verdict token parses.

    For more information, see:

    - [kanana-safeguard-8b](https://huggingface.co/kakaocorp/kanana-safeguard-8b) (default).
    - [kanana-safeguard-siren-8b](https://huggingface.co/kakaocorp/kanana-safeguard-siren-8b).
    - [kanana-safeguard-prompt-2.1b](https://huggingface.co/kakaocorp/kanana-safeguard-prompt-2.1b).

    Args:
        model_id: Optional HuggingFace model ID; one of ``SUPPORTED_MODELS``. Defaults to
            ``kakaocorp/kanana-safeguard-8b``. The choice of model selects the taxonomy
            (see above).
        provider: Optional pre-configured provider. Defaults to a ``HuggingFaceProvider``
            loading a causal LM (``AutoModelForCausalLM`` + ``AutoTokenizer``).

    """

    SUPPORTED_MODELS: ClassVar = [
        "kakaocorp/kanana-safeguard-8b",
        "kakaocorp/kanana-safeguard-siren-8b",
        "kakaocorp/kanana-safeguard-prompt-2.1b",
    ]

    METADATA: ClassVar[GuardrailMetadata] = GUARDRAIL_METADATA[GuardrailName.KANANA_SAFEGUARD]

    def __init__(self, model_id: str | None = None, provider: StandardProvider | None = None) -> None:
        """Initialize the Kanana Safeguard guardrail.

        Args:
            model_id: Optional HuggingFace model ID; one of ``SUPPORTED_MODELS``
                (``kakaocorp/kanana-safeguard-8b``, ``kakaocorp/kanana-safeguard-siren-8b``,
                ``kakaocorp/kanana-safeguard-prompt-2.1b``). Defaults to the harm model
                ``kakaocorp/kanana-safeguard-8b``. Each variant carries its own unsafe-code
                taxonomy (see the class docstring).
            provider: Optional pre-configured provider (e.g. a ``LlamafileProvider`` or a
                customized ``HuggingFaceProvider``). Defaults to a ``HuggingFaceProvider``;
                when a ``HuggingFaceProvider`` is used, the causal-LM loader classes
                (``AutoModelForCausalLM`` + ``AutoTokenizer``) are enforced at load time.

        Raises:
            ValueError: If ``model_id`` is not in ``SUPPORTED_MODELS``.

        """
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
        """Classify ``input_text`` (and, for the harm model, an assistant ``output_text``).

        Args:
            input_text: The user turn to classify. Must be a single string — list (batch)
                inputs are not supported. Korean is the primary training language.
            output_text: Optional assistant response. Only the harm model
                (``kakaocorp/kanana-safeguard-8b``) is trained to judge an assistant turn;
                the ``-siren-8b`` and ``-prompt-2.1b`` variants silently ignore this
                argument.
            **kwargs: Forwarded to the base ``validate`` implementation.

        Returns:
            A :class:`GuardrailOutput` where ``valid=True`` on a ``<SAFE>`` verdict.
            On ``<UNSAFE-*>``, ``categories`` holds the matched code (triggered) and
            ``extra["verdict"]`` the raw token. Fails closed (``valid=False`` with
            ``extra={"parse_failure": True}``) when no verdict token parses.

        Raises:
            TypeError: If a list input reaches this guardrail (single strings only).

        """
        result = super().validate(input_text, output_text=output_text, **kwargs)
        if isinstance(result, list):
            msg = "KananaSafeguard.validate received a list input but only supports single strings."
            raise TypeError(msg)
        return result

    def _pre_processing(
        self, input_text: str, output_text: str | None = None, **kwargs: Any
    ) -> GuardrailPreprocessOutput[KananaPreprocessData]:
        """Shape the chat messages: a user turn, plus an assistant turn for the harm model.

        Args:
            input_text: The user turn to classify.
            output_text: Optional assistant response; appended as an assistant message only
                when the loaded model is the harm variant (``kakaocorp/kanana-safeguard-8b``).
            **kwargs: Ignored.

        Returns:
            A :class:`GuardrailPreprocessOutput` containing the ``messages`` list.

        """
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

    def _post_processing(self, model_outputs: GuardrailInferenceOutput[KananaInferenceData]) -> GuardrailOutput:
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
