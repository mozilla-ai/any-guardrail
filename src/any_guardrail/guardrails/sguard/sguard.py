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

SguardPreprocessData = AnyDict
SguardInferenceData = AnyDict

CONTENT_FILTER_MODEL = "SamsungSDS-Research/SGuard-ContentFilter-2B-v1"
JAILBREAK_FILTER_MODEL = "SamsungSDS-Research/SGuard-JailbreakFilter-2B-v1"

# ContentFilter taxonomy (MLCommons-aligned, 5 categories).
SGUARD_CONTENT_CATEGORIES = ["Crime", "Manipulation", "Privacy", "Sexual", "Violence"]

MAX_NEW_TOKENS = 16
_UNSAFE = re.compile(r"\bunsafe\b", re.IGNORECASE)


class Sguard(ThreeStageGuardrail[SguardPreprocessData, SguardInferenceData]):
    """Samsung SDS SGuard-v1 — multilingual content-safety and jailbreak filters.

    Two 2B Granite-based guards selectable by ``model_id``:

    - **ContentFilter** classifies prompt/response against five categories
      (Crime, Manipulation, Privacy, Sexual, Violence).
    - **JailbreakFilter** flags jailbreak-framed inputs (binary safe/unsafe).

    ``valid`` is ``True`` when nothing is flagged. Fails closed (``valid=False`` with
    ``extra={"parse_failure": True}``) when the output cannot be read.

    Note: SGuard natively emits per-category safe/unsafe *tokens* read from logits; this
    integration parses the decoded text and is best-effort — validate against the model
    on real weights before production use.

    For more information, see the model cards:

    - [SGuard-ContentFilter-2B-v1](https://huggingface.co/SamsungSDS-Research/SGuard-ContentFilter-2B-v1) (default).
    - [SGuard-JailbreakFilter-2B-v1](https://huggingface.co/SamsungSDS-Research/SGuard-JailbreakFilter-2B-v1).

    Args:
        model_id: Optional HuggingFace model ID. Defaults to the ContentFilter model.
        provider: Optional pre-configured provider. Defaults to a ``HuggingFaceProvider``
            loading a causal LM.
    """

    SUPPORTED_MODELS: ClassVar = [CONTENT_FILTER_MODEL, JAILBREAK_FILTER_MODEL]

    def __init__(self, model_id: str | None = None, provider: StandardProvider | None = None) -> None:
        """Initialize the SGuard guardrail."""
        self.model_id = default(model_id, self.SUPPORTED_MODELS)
        self.is_content_filter = self.model_id == CONTENT_FILTER_MODEL
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
            msg = "Sguard.validate received a list input but only supports single strings."
            raise TypeError(msg)
        return result

    def _pre_processing(
        self, input_text: str, output_text: str | None = None, **kwargs: Any
    ) -> GuardrailPreprocessOutput[SguardPreprocessData]:
        del kwargs
        messages: ChatMessages = [{"role": "user", "content": input_text}]
        if output_text is not None:
            messages.append({"role": "assistant", "content": output_text})
        return GuardrailPreprocessOutput(data={"messages": messages})

    def _inference(
        self, model_inputs: GuardrailPreprocessOutput[SguardPreprocessData]
    ) -> GuardrailInferenceOutput[SguardInferenceData]:
        return self.provider.generate_chat(
            messages=model_inputs.data["messages"], max_new_tokens=MAX_NEW_TOKENS, do_sample=False
        )

    def _post_processing(
        self, model_outputs: GuardrailInferenceOutput[SguardInferenceData]
    ) -> GuardrailOutput:
        text = model_outputs.data["generated_text"]
        usage = GuardrailUsage(
            prompt_tokens=model_outputs.data.get("prompt_token_count"),
            completion_tokens=model_outputs.data.get("completion_token_count"),
        )
        if not text.strip():
            return GuardrailOutput(valid=False, explanation=text, extra={"parse_failure": True})
        if self.is_content_filter:
            categories = [
                CategoryResult(name=name, triggered=bool(re.search(rf"{name}\s*[:\-]?\s*unsafe", text, re.IGNORECASE)))
                for name in SGUARD_CONTENT_CATEGORIES
            ]
            flagged = any(category.triggered for category in categories) or bool(_UNSAFE.search(text))
            return GuardrailOutput(valid=not flagged, explanation=text, categories=categories, usage=usage)
        unsafe = bool(_UNSAFE.search(text))
        return GuardrailOutput(
            valid=not unsafe,
            explanation=text,
            categories=[CategoryResult(name="jailbreak", triggered=unsafe)],
            usage=usage,
        )
