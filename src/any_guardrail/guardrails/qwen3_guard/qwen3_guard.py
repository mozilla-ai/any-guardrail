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

Qwen3GuardPreprocessData = AnyDict
Qwen3GuardInferenceData = AnyDict

# Qwen3Guard safety taxonomy; the model reports these names verbatim on its
# ``Categories:`` line ("Jailbreak" applies to prompt moderation only).
QWEN3GUARD_CATEGORIES = [
    "Violent",
    "Non-violent Illegal Acts",
    "Sexual Content or Sexual Acts",
    "PII",
    "Suicide & Self-Harm",
    "Unethical Acts",
    "Politically Sensitive Topics",
    "Copyright Violation",
    "Jailbreak",
]

# Canonical risk mapping for the three severity levels.
SEVERITY_RISK = {"Safe": 0.0, "Controversial": 0.5, "Unsafe": 1.0}

MAX_NEW_TOKENS = 128

_SAFETY = re.compile(r"Safety:\s*(Safe|Unsafe|Controversial)", re.IGNORECASE)
_CATEGORIES_LINE = re.compile(r"Categories:\s*(.+)", re.IGNORECASE)
# Deliberately case-sensitive and in model-card order so "Violent" cannot
# match inside "Non-violent Illegal Acts".
_CATEGORY = re.compile("|".join(re.escape(category) for category in QWEN3GUARD_CATEGORIES))
_REFUSAL = re.compile(r"Refusal:\s*(Yes|No)", re.IGNORECASE)
_THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)


class Qwen3Guard(ThreeStageGuardrail[Qwen3GuardPreprocessData, Qwen3GuardInferenceData]):
    """Qwen3Guard-Gen — generative safety moderation with three-level severity across 119 languages (Qwen).

    Decoder LLM (Apache-2.0) whose chat template embeds the safety-classifier
    instruction: the user prompt alone triggers prompt moderation; supplying an
    assistant ``output_text`` switches to response moderation. The model reports a
    severity (``Safe`` / ``Controversial`` / ``Unsafe``, where ``Controversial``
    means harmfulness is context-dependent), the violated categories from a
    nine-item taxonomy (Violent, Non-violent Illegal Acts, Sexual Content or
    Sexual Acts, PII, Suicide & Self-Harm, Unethical Acts, Politically Sensitive
    Topics, Copyright Violation, plus Jailbreak for prompt moderation), and — in
    response mode — whether the response is a refusal.

    ``GuardrailOutput`` mapping: ``valid`` is ``True`` only for ``Safe`` verdicts
    (``Controversial`` also passes when ``strict=False``); ``score`` maps the
    severity onto the canonical risk axis, higher = riskier (Safe 0.0,
    Controversial 0.5, Unsafe 1.0); ``categories`` holds one triggered entry per
    reported category (plus a ``refusal`` entry in response mode);
    ``extra["severity"]`` carries the verbatim severity and ``explanation`` the
    full generation. Fails closed (``valid=False`` with
    ``extra={"parse_failure": True}``) when no severity parses. For the
    token-level streaming variants (``Qwen3Guard-Stream-*``), see
    ``Qwen3GuardStream``.

    For more information, see:

    - [Qwen3Guard-Gen-0.6B model card](https://huggingface.co/Qwen/Qwen3Guard-Gen-0.6B) (default).
    - [Qwen3Guard-Gen-4B model card](https://huggingface.co/Qwen/Qwen3Guard-Gen-4B).
    - [Qwen3Guard-Gen-8B model card](https://huggingface.co/Qwen/Qwen3Guard-Gen-8B).
    - [Qwen3Guard Technical Report](https://arxiv.org/abs/2510.14276).

    Args:
        strict: If ``True`` (default), only ``Safe`` verdicts pass validation; set
            ``False`` to let ``Controversial`` content pass (``valid=True``), leaving
            it reflected only in ``score`` and ``extra["severity"]``.
        model_id: Optional HuggingFace model ID, one of ``SUPPORTED_MODELS``.
            Defaults to ``Qwen/Qwen3Guard-Gen-0.6B``.
        provider: Optional pre-configured provider. Defaults to a ``HuggingFaceProvider``
            loading a causal LM.

    """

    SUPPORTED_MODELS: ClassVar = [
        "Qwen/Qwen3Guard-Gen-0.6B",
        "Qwen/Qwen3Guard-Gen-4B",
        "Qwen/Qwen3Guard-Gen-8B",
    ]

    def __init__(
        self,
        strict: bool = True,
        model_id: str | None = None,
        provider: StandardProvider | None = None,
    ) -> None:
        """Initialize the Qwen3Guard guardrail."""
        self.model_id = default(model_id, self.SUPPORTED_MODELS)
        self.strict = strict
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
        """Moderate ``input_text`` (or, when ``output_text`` is given, the assistant response to it)."""
        result = super().validate(input_text, output_text=output_text, **kwargs)
        if isinstance(result, list):
            msg = "Qwen3Guard.validate received a list input but only supports single strings."
            raise TypeError(msg)
        return result

    def _pre_processing(
        self, input_text: str, output_text: str | None = None, **kwargs: Any
    ) -> GuardrailPreprocessOutput[Qwen3GuardPreprocessData]:
        del kwargs
        messages: ChatMessages = [{"role": "user", "content": input_text}]
        if output_text is not None:
            messages.append({"role": "assistant", "content": output_text})
        return GuardrailPreprocessOutput(data={"messages": messages, "has_response": output_text is not None})

    def _inference(
        self, model_inputs: GuardrailPreprocessOutput[Qwen3GuardPreprocessData]
    ) -> GuardrailInferenceOutput[Qwen3GuardInferenceData]:
        result = self.provider.generate_chat(
            messages=model_inputs.data["messages"], max_new_tokens=MAX_NEW_TOKENS, do_sample=False
        )
        # Carry has_response through so _post_processing surfaces the Refusal field only in response mode.
        result.data["has_response"] = model_inputs.data["has_response"]
        return result

    def _post_processing(self, model_outputs: GuardrailInferenceOutput[Qwen3GuardInferenceData]) -> GuardrailOutput:
        text = model_outputs.data["generated_text"]
        has_response = model_outputs.data.get("has_response", False)
        without_think = _THINK_PATTERN.sub("", text).strip()
        safety = _SAFETY.search(without_think)
        if safety is None:
            return GuardrailOutput(valid=False, explanation=text, extra={"parse_failure": True})
        severity = safety.group(1).capitalize()
        categories: list[CategoryResult] = []
        categories_line = _CATEGORIES_LINE.search(without_think)
        if categories_line is not None:
            categories = [
                CategoryResult(name=name, triggered=True)
                for name in dict.fromkeys(_CATEGORY.findall(categories_line.group(1)))
            ]
        if has_response:
            # Refusal is auxiliary metadata about the judged response; a missing line
            # is tolerated (triggered=None) rather than failing closed — the Safety
            # line above already carries the verdict.
            refusal = _REFUSAL.search(without_think)
            triggered = refusal.group(1).lower() == "yes" if refusal else None
            categories.append(CategoryResult(name="refusal", triggered=triggered))
        return GuardrailOutput(
            valid=severity == "Safe" if self.strict else severity != "Unsafe",
            explanation=text,
            score=SEVERITY_RISK[severity],
            categories=categories,
            extra={"severity": severity},
            usage=GuardrailUsage(
                prompt_tokens=model_outputs.data.get("prompt_token_count"),
                completion_tokens=model_outputs.data.get("completion_token_count"),
            ),
        )
