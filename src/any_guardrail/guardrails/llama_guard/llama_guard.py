import re
from typing import Any, ClassVar

from any_guardrail.base import GuardrailName, GuardrailOutput, ThreeStageGuardrail
from any_guardrail.providers.base import StandardProvider
from any_guardrail.providers.huggingface import HuggingFaceProvider
from any_guardrail.registry import GUARDRAIL_METADATA
from any_guardrail.taxonomy import GuardrailMetadata
from any_guardrail.types import (
    AnyDict,
    CategoryResult,
    GuardrailInferenceOutput,
    GuardrailPreprocessOutput,
    GuardrailUsage,
)

LlamaGuardPreprocessData = AnyDict  # {"messages": list, "chat_template_kwargs": dict}
LlamaGuardInferenceData = AnyDict  # {"generated_text": str, ...} (shape from provider.generate_chat)

LLAMA_GUARD_CATEGORIES = {
    "S1": "Violent Crimes",
    "S2": "Non-Violent Crimes",
    "S3": "Sex-Related Crimes",
    "S4": "Child Sexual Exploitation",
    "S5": "Defamation",
    "S6": "Specialized Advice",
    "S7": "Privacy",
    "S8": "Intellectual Property",
    "S9": "Indiscriminate Weapons",
    "S10": "Hate",
    "S11": "Suicide & Self-Harm",
    "S12": "Sexual Content",
    "S13": "Elections",
    "S14": "Code Interpreter Abuse",
}
"""MLCommons-aligned hazard taxonomy used by Llama Guard 3 and 4."""

_CATEGORY_CODE_PATTERN = re.compile(r"\bS(\d{1,2})\b")


def _parse_violated_categories(generated_text: str) -> list[CategoryResult]:
    r"""Extract violated S-codes (e.g. ``unsafe\nS1,S10``) as CategoryResults.

    Codes are deduplicated in order of first appearance. Unknown codes are kept
    with ``description=None`` so future taxonomy additions still surface.
    """
    seen: dict[str, None] = {}
    for match in _CATEGORY_CODE_PATTERN.finditer(generated_text):
        seen.setdefault(f"S{match.group(1)}")
    return [CategoryResult(name=code, description=LLAMA_GUARD_CATEGORIES.get(code), triggered=True) for code in seen]


class LlamaGuard(ThreeStageGuardrail[LlamaGuardPreprocessData, LlamaGuardInferenceData]):
    """Llama Guard — decoder-LLM safety classifier judging prompts and responses against the 14-category MLCommons hazard taxonomy (Meta).

    Llama Guard is Meta's instruction-tuned safety model. Each call wraps a user prompt (and,
    optionally, an assistant response) in the model's moderation template listing the 14 MLCommons
    hazard categories (``S1`` Violent Crimes ... ``S14`` Code Interpreter Abuse), then generates a
    verdict: ``safe``, or ``unsafe`` followed by the violated category codes. This wrapper supports
    both Llama Guard 3 (1B / 8B, text-only) and Llama Guard 4 (12B, natively multimodal — this
    integration passes text content only); the per-variant chat-template quirks (v3 evaluates the
    conversation as-is without an appended assistant prefix, v4 uses the standard template) are
    handled internally, so callers select a variant purely via ``model_id``. Llama Guard 3 is
    trained for multilingual moderation across eight languages (English, French, German, Hindi,
    Italian, Portuguese, Spanish, Thai).

    Verdict mapping onto ``GuardrailOutput``:

    - ``valid`` is ``False`` when the generation contains ``unsafe`` (case-insensitive), ``True``
      otherwise.
    - ``categories`` lists the violated hazard codes parsed from the generation, one
      ``CategoryResult`` per code (``name`` = ``Sx``, ``description`` = the taxonomy label,
      ``triggered=True``), deduplicated in order of first appearance; empty when the verdict is
      ``safe``. Unknown codes are kept with ``description=None`` so taxonomy additions still surface.
    - ``explanation`` is the raw generated text.
    - ``usage`` carries the prompt / completion token counts.
    - No canonical ``score`` and no ``spans`` are produced (``score`` is ``None``).

    Expected inputs: a single ``input_text`` string (the user prompt) plus an optional
    ``output_text`` string (an assistant response). When ``output_text`` is supplied, the model
    judges the full ``[user, assistant]`` turn — i.e. it moderates the response in the context of
    the prompt. Single strings only; list / batch input is not supported.

    The models are gated on HuggingFace and distributed under Meta's Llama Community License.

    For more information, see:

    - [Llama Guard 3 model card (Meta)](https://www.llama.com/docs/model-cards-and-prompt-formats/llama-guard-3/)
    - [Llama Guard 4 model card (Meta)](https://www.llama.com/docs/model-cards-and-prompt-formats/llama-guard-4/)
    - [meta-llama/Llama-Guard-3-1B](https://huggingface.co/meta-llama/Llama-Guard-3-1B)
    - [meta-llama/Llama-Guard-3-8B](https://huggingface.co/meta-llama/Llama-Guard-3-8B)
    - [meta-llama/Llama-Guard-4-12B](https://huggingface.co/meta-llama/Llama-Guard-4-12B)

    Args:
        model_id: Optional HuggingFace model ID; must be one of ``SUPPORTED_MODELS``. Defaults to
            ``meta-llama/Llama-Guard-3-1B`` (Llama Guard 3). Pass ``meta-llama/Llama-Guard-4-12B``
            to select the Llama Guard 4 variant.
        provider: Optional pre-configured provider. Defaults to a ``HuggingFaceProvider`` loading
            the right head for the variant (a causal LM for v3, ``Llama4ForConditionalGeneration``
            for v4). A supplied ``HuggingFaceProvider`` is corrected to the same classes at load
            time; a non-HF provider (e.g. ``LlamafileProvider``) is used as-is.

    """

    SUPPORTED_MODELS: ClassVar = [
        "meta-llama/Llama-Guard-3-1B",
        "meta-llama/Llama-Guard-3-8B",
        "meta-llama/Llama-Guard-4-12B",
    ]

    METADATA: ClassVar[GuardrailMetadata] = GUARDRAIL_METADATA[GuardrailName.LLAMA_GUARD]

    def __init__(
        self,
        model_id: str | None = None,
        provider: StandardProvider | None = None,
    ) -> None:
        """Initialize the Llama Guard guardrail.

        Args:
            model_id: Optional HuggingFace model ID; must be one of ``SUPPORTED_MODELS``. Selects
                the variant — ``meta-llama/Llama-Guard-3-1B`` (default) or
                ``meta-llama/Llama-Guard-3-8B`` for Llama Guard 3, ``meta-llama/Llama-Guard-4-12B``
                for Llama Guard 4.
            provider: Optional pre-configured provider. When ``None``, a ``HuggingFaceProvider`` is
                built with the correct model / tokenizer classes for the selected variant (a causal
                LM for v3; ``Llama4ForConditionalGeneration`` + ``AutoProcessor`` for v4). A
                supplied ``HuggingFaceProvider`` is corrected to the same classes at load time
                (without mutating it); any other provider (e.g. ``LlamafileProvider``) is used
                as-is.

        Raises:
            ValueError: If ``model_id`` is not one of ``SUPPORTED_MODELS``.

        """
        self.model_id = model_id or self.SUPPORTED_MODELS[0]

        # Determine per-variant chat-template behavior up-front so this is the only
        # place that knows the v3-vs-v4 quirk, regardless of which provider is used.
        if self._is_version_4:
            # v4 wants the standard "add the assistant prefix" template behavior;
            # provider.generate_chat already defaults to that, so no override.
            self._chat_template_kwargs: AnyDict = {}
        elif self.model_id in self.SUPPORTED_MODELS:
            # Llama Guard 3 expects to evaluate the conversation as-is, without an
            # appended assistant prefix.
            self._chat_template_kwargs = {"add_generation_prompt": False}
        else:
            msg = f"Unsupported model_id: {self.model_id}"
            raise ValueError(msg)

        # Lazy-import transformers so users on `any-guardrail[llamafile]`
        # (without the huggingface extra) can construct LlamaGuard with a
        # non-HF provider (e.g. LlamafileProvider) without paying the import
        # cost or hitting ImportError at module load time.
        load_kwargs: AnyDict = {}
        if provider is not None:
            self.provider = provider
            if isinstance(self.provider, HuggingFaceProvider):
                # Llama Guard is a causal LM (or multimodal seq2seq for v4). A
                # default-constructed HuggingFaceProvider targets
                # AutoModelForSequenceClassification, which would silently load
                # the wrong head. Enforce the right classes for this load
                # (does not mutate provider state).
                if self._is_version_4:
                    from transformers import AutoProcessor, Llama4ForConditionalGeneration

                    load_kwargs = {
                        "model_class": Llama4ForConditionalGeneration,
                        "tokenizer_class": AutoProcessor,
                    }
                else:
                    from transformers import AutoModelForCausalLM, AutoTokenizer

                    load_kwargs = {"model_class": AutoModelForCausalLM, "tokenizer_class": AutoTokenizer}
        else:
            from transformers import (
                AutoModelForCausalLM,
                AutoProcessor,
                AutoTokenizer,
                Llama4ForConditionalGeneration,
            )

            if self._is_version_4:
                self.provider = HuggingFaceProvider(
                    model_class=Llama4ForConditionalGeneration,
                    tokenizer_class=AutoProcessor,
                )
            else:
                self.provider = HuggingFaceProvider(
                    model_class=AutoModelForCausalLM,
                    tokenizer_class=AutoTokenizer,
                )
        self.provider.load_model(self.model_id, **load_kwargs)

    def _build_conversation(self, input_text: str, output_text: str | None) -> list[AnyDict]:
        """Shape the chat conversation per model variant."""
        uses_multimodal_content = self.model_id == self.SUPPORTED_MODELS[0] or self._is_version_4
        if uses_multimodal_content:
            user_turn: AnyDict = {"role": "user", "content": [{"type": "text", "text": input_text}]}
        else:
            user_turn = {"role": "user", "content": input_text}
        conversation: list[AnyDict] = [user_turn]
        if output_text:
            if uses_multimodal_content:
                conversation.append({"role": "assistant", "content": [{"type": "text", "text": output_text}]})
            else:
                conversation.append({"role": "assistant", "content": output_text})
        return conversation

    def _pre_processing(
        self, input_text: str, output_text: str | None = None, **kwargs: Any
    ) -> GuardrailPreprocessOutput[LlamaGuardPreprocessData]:
        """Build the moderation conversation and per-variant chat-template kwargs.

        Args:
            input_text: The user prompt to moderate, placed in the ``user`` turn of the
                conversation.
            output_text: Optional assistant response. When provided, an ``assistant`` turn is
                appended so Llama Guard judges the response in the context of the prompt.
            **kwargs: Extra chat-template arguments merged over the variant defaults (e.g.
                ``add_generation_prompt``), forwarded to ``provider.generate_chat``.

        Returns:
            GuardrailPreprocessOutput wrapping ``{"messages": ..., "chat_template_kwargs": ...}``.

        """
        conversation = self._build_conversation(input_text, output_text)
        chat_template_kwargs: AnyDict = {**self._chat_template_kwargs, **kwargs}
        return GuardrailPreprocessOutput(
            data={
                "messages": conversation,
                "chat_template_kwargs": chat_template_kwargs,
            }
        )

    def _inference(
        self, model_inputs: GuardrailPreprocessOutput[LlamaGuardPreprocessData]
    ) -> GuardrailInferenceOutput[LlamaGuardInferenceData]:
        """Dispatch to ``provider.generate_chat`` with version-appropriate gen params."""
        max_new_tokens = 10 if self._is_version_4 else 20
        # Llama Guard 3 was historically generated with ``pad_token_id=0``; preserve
        # that to keep generation behavior bit-identical with the pre-refactor path.
        generation_kwargs: AnyDict | None = None if self._is_version_4 else {"pad_token_id": 0}
        return self.provider.generate_chat(
            messages=model_inputs.data["messages"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            chat_template_kwargs=model_inputs.data["chat_template_kwargs"] or None,
            generation_kwargs=generation_kwargs,
        )

    def _post_processing(self, model_outputs: GuardrailInferenceOutput[LlamaGuardInferenceData]) -> GuardrailOutput:
        generated_text: str = model_outputs.data["generated_text"]
        unsafe = "unsafe" in generated_text.lower()
        return GuardrailOutput(
            valid=not unsafe,
            explanation=generated_text,
            categories=_parse_violated_categories(generated_text) if unsafe else [],
            usage=GuardrailUsage(
                prompt_tokens=model_outputs.data.get("prompt_token_count"),
                completion_tokens=model_outputs.data.get("completion_token_count"),
            ),
        )

    @property
    def _is_version_4(self) -> bool:
        return self.model_id == self.SUPPORTED_MODELS[-1]
