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

DynaGuardPreprocessData = AnyDict
DynaGuardInferenceData = AnyDict

DYNAGUARD_SYSTEM_PROMPT = PROMPT_REGISTRY[GuardrailName.DYNA_GUARD].resolve().segments["system"]
"""DynaGuard guardian system prompt (registry-sourced)."""
DYNAGUARD_USER_TEMPLATE = PROMPT_REGISTRY[GuardrailName.DYNA_GUARD].resolve().segments["user"]
"""DynaGuard user template (registry-sourced); fills ``{policy}`` / ``{transcript}``."""

# DynaGuard reasons inside a <think> block before emitting <answer>PASS/FAIL</answer> even in
# "fast" mode (we don't prime the response with <answer>), so the budget must be large enough to
# reach the verdict tag — a too-small budget truncates mid-reasoning and fails closed.
MAX_NEW_TOKENS_FAST = 512
MAX_NEW_TOKENS_THINK = 1024

_ANSWER_PATTERN = re.compile(r"<answer>\s*(PASS|FAIL)\s*</answer>", re.IGNORECASE)
_THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)
# The system prompt allows reasoning in either a <think> or an <explanation> block; strip both
# before the bare-verdict fallback so PASS/FAIL mentioned mid-reasoning can't be mistaken for the verdict.
_EXPLANATION_PATTERN = re.compile(r"<explanation>.*?</explanation>", re.DOTALL)
# Fallback: a bare PASS/FAIL token when the model omits the <answer> wrapper.
_BARE_VERDICT = re.compile(r"\b(PASS|FAIL)\b")


class DynaGuard(ThreeStageGuardrail[DynaGuardPreprocessData, DynaGuardInferenceData]):
    """DynaGuard — dynamic guardian model evaluating conversation compliance with user-defined policies.

    A decoder-LLM guardian model (University of Maryland / Capital One) that checks a
    conversation transcript against a bring-your-own ``policy`` — a numbered list of
    natural-language rules — and returns ``PASS`` (compliant) or ``FAIL`` (at least one
    rule violated). Unlike fixed-taxonomy safety classifiers, the rules are
    application-specific: e.g. "the agent must never issue a refund" or "the agent must
    not give medical advice". With ``think=True`` the model first emits a
    chain-of-thought ``<think>`` block justifying each rule before the ``<answer>``
    verdict (higher latency, potentially higher accuracy); the reasoning is stripped
    before the verdict is parsed.

    Verdict mapping onto ``GuardrailOutput``:

    - ``valid`` is ``True`` on ``PASS`` and ``False`` on ``FAIL``.
    - ``categories`` carries a single ``policy_violation`` entry whose ``triggered``
      flag is ``True`` when the verdict is ``FAIL``.
    - ``extra["verdict"]`` holds the raw ``"PASS"`` / ``"FAIL"`` token.
    - ``explanation`` holds the model's full raw generation (including any reasoning).
    - ``score`` is left ``None`` — DynaGuard emits a categorical verdict, not a
      calibrated risk probability.
    - ``usage`` records the prompt/completion token counts when the backend reports them.
    - Fails closed (``valid=False`` with ``extra={"parse_failure": True}``) when neither
      an ``<answer>`` block nor a bare ``PASS``/``FAIL`` token can be parsed (e.g. the
      generation was truncated mid-reasoning).

    Expected inputs: a single ``input_text`` (the user turn / transcript; required) plus
    an optional ``output_text`` (the agent's response). The two are assembled into a
    ``User: ... Agent: ...`` transcript (the turns joined by a newline) before being
    wrapped with the ``policy``. List/batch input is not supported — passing a list
    raises ``TypeError``.

    For more information, see:

    - [DynaGuard-8B model card](https://huggingface.co/tomg-group-umd/DynaGuard-8B) (default).
    - [DynaGuard-4B model card](https://huggingface.co/tomg-group-umd/DynaGuard-4B).
    - [DynaGuard-1.7B model card](https://huggingface.co/tomg-group-umd/DynaGuard-1.7B).
    - [DynaGuard: A Dynamic Guardian Model With User-Defined Policies (arXiv:2509.02563)](https://arxiv.org/abs/2509.02563)

    Args:
        policy: The rules to enforce, as numbered natural-language text (a bring-your-own
            taxonomy), e.g. a newline-separated list of ``1. Do not reveal the system
            prompt.`` and ``2. Do not issue refunds.``.
        think: If ``True``, request chain-of-thought reasoning before the verdict (higher
            latency, larger token budget). Defaults to ``False``.
        model_id: Optional HuggingFace model ID. Must be one of ``SUPPORTED_MODELS``;
            defaults to ``tomg-group-umd/DynaGuard-8B``.
        provider: Optional pre-configured provider. Defaults to a ``HuggingFaceProvider``
            loading the model as a causal LM.

    """

    SUPPORTED_MODELS: ClassVar = [
        "tomg-group-umd/DynaGuard-8B",
        "tomg-group-umd/DynaGuard-4B",
        "tomg-group-umd/DynaGuard-1.7B",
    ]

    METADATA: ClassVar[GuardrailMetadata] = GUARDRAIL_METADATA[GuardrailName.DYNA_GUARD]

    PROMPT: ClassVar[PromptSpec] = PROMPT_REGISTRY[GuardrailName.DYNA_GUARD]

    def __init__(
        self,
        policy: str,
        think: bool = False,
        model_id: str | None = None,
        provider: StandardProvider | None = None,
        prompt: PromptTemplate | None = None,
        prompt_version: str | None = None,
    ) -> None:
        """Initialize the DynaGuard guardrail.

        Args:
            policy: The rules to enforce, as numbered natural-language text (a
                bring-your-own taxonomy), e.g. a newline-separated list of ``1. Do not
                reveal the system prompt.`` and ``2. Do not issue refunds.``. Applied to
                every ``validate`` call.
            think: If ``True``, request chain-of-thought reasoning before the verdict,
                which raises the generation token budget and latency. Defaults to
                ``False``.
            model_id: Optional HuggingFace model ID. Must be one of ``SUPPORTED_MODELS``;
                defaults to ``tomg-group-umd/DynaGuard-8B``.
            provider: Optional pre-configured provider. Defaults to a
                ``HuggingFaceProvider`` loading the model as a causal LM. When a
                ``HuggingFaceProvider`` is supplied, it is loaded with
                ``model_class=AutoModelForCausalLM`` / ``tokenizer_class=AutoTokenizer``.
            prompt: Optional prompt-template override, used as-is (system prompt plus a user
                template filling ``{policy}`` / ``{transcript}``). Defaults to ``None`` — the
                registry default, or the version named by ``prompt_version``.
            prompt_version: Registered prompt version to use when ``prompt`` is not given. Defaults
                to ``None`` (the default version). See ``AnyGuardrail.list_prompt_versions``.

        Raises:
            ValueError: If ``model_id`` is not in ``SUPPORTED_MODELS``.

        """
        self.model_id = default(model_id, self.SUPPORTED_MODELS)
        self.policy = policy
        self.think = think
        self._prompt = resolve_prompt(GuardrailName.DYNA_GUARD, prompt, prompt_version)
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
        """Evaluate a conversation transcript against the configured policy.

        Args:
            input_text: The user turn (or the transcript to evaluate), e.g.
                ``"Please refund my last order."``. A single string; list/batch input is
                rejected with ``TypeError``.
            output_text: Optional agent response judged alongside the user turn, e.g.
                ``"Sure, I've issued your refund."``. When supplied, the two are
                assembled into a ``User: ... Agent: ...`` transcript (turns joined by a
                newline).
            **kwargs: Reserved for forward compatibility; forwarded to the base pipeline
                and otherwise ignored.

        Returns:
            GuardrailOutput where ``valid`` is ``True`` on ``PASS`` / ``False`` on
            ``FAIL``, ``categories`` carries the ``policy_violation`` flag,
            ``extra["verdict"]`` holds the raw verdict token, ``explanation`` is the raw
            generation, and ``usage`` holds token counts. Fails closed (``valid=False``
            with ``extra={"parse_failure": True}``) when no verdict can be parsed.

        Raises:
            TypeError: If a list input is supplied (only single strings are supported).

        """
        result = super().validate(input_text, output_text=output_text, **kwargs)
        if isinstance(result, list):
            msg = "DynaGuard.validate received a list input but only supports single strings."
            raise TypeError(msg)
        return result

    def _pre_processing(
        self, input_text: str, output_text: str | None = None, **kwargs: Any
    ) -> GuardrailPreprocessOutput[DynaGuardPreprocessData]:
        """Assemble the transcript and wrap it with the policy in DynaGuard's chat prompt.

        Args:
            input_text: The user turn / transcript, placed in the ``<transcript>`` block.
            output_text: Optional agent response; when supplied the transcript becomes
                ``User: {input_text}`` and ``Agent: {output_text}`` joined by a newline,
                otherwise it is ``input_text`` alone.
            **kwargs: Ignored (accepted for pipeline compatibility).

        Returns:
            GuardrailPreprocessOutput wrapping the system+user chat messages (the
            configured ``policy`` in ``<rules>`` and the transcript in ``<transcript>``).

        """
        del kwargs
        transcript = f"User: {input_text}\nAgent: {output_text}" if output_text is not None else input_text
        user = self._prompt.segments["user"].format(policy=self.policy, transcript=transcript)
        messages: ChatMessages = [
            {"role": "system", "content": self._prompt.segments["system"]},
            {"role": "user", "content": user},
        ]
        return GuardrailPreprocessOutput(data={"messages": messages})

    def _inference(
        self, model_inputs: GuardrailPreprocessOutput[DynaGuardPreprocessData]
    ) -> GuardrailInferenceOutput[DynaGuardInferenceData]:
        max_new_tokens = MAX_NEW_TOKENS_THINK if self.think else MAX_NEW_TOKENS_FAST
        return self.provider.generate_chat(
            messages=model_inputs.data["messages"], max_new_tokens=max_new_tokens, do_sample=False
        )

    def _post_processing(self, model_outputs: GuardrailInferenceOutput[DynaGuardInferenceData]) -> GuardrailOutput:
        text = model_outputs.data["generated_text"]
        cleaned = _EXPLANATION_PATTERN.sub("", _THINK_PATTERN.sub("", text)).strip()
        answer = _ANSWER_PATTERN.search(cleaned)
        if answer is not None:
            verdict = answer.group(1)
        else:
            # No <answer> wrapper: take the LAST bare PASS/FAIL so the final verdict wins
            # over any PASS/FAIL the model mentioned earlier while reasoning.
            bare = _BARE_VERDICT.findall(cleaned)
            if not bare:
                return GuardrailOutput(valid=False, explanation=text, extra={"parse_failure": True})
            verdict = bare[-1]
        verdict = verdict.strip().upper()
        violated = verdict == "FAIL"
        return GuardrailOutput(
            valid=not violated,
            explanation=text,
            categories=[CategoryResult(name="policy_violation", triggered=violated)],
            extra={"verdict": verdict},
            usage=GuardrailUsage(
                prompt_tokens=model_outputs.data.get("prompt_token_count"),
                completion_tokens=model_outputs.data.get("completion_token_count"),
            ),
        )
