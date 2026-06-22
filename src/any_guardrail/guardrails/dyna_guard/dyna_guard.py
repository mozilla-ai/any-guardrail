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

DynaGuardPreprocessData = AnyDict
DynaGuardInferenceData = AnyDict

DYNAGUARD_SYSTEM_PROMPT = (
    "You are a guardian model evaluating the compliance of an agent with a list of rules.\n"
    "The rules will be provided in a numbered format, starting with a <rules> tag and ending "
    "with a </rules> tag.\n"
    "You will be given the transcript that contains output from the agent, starting with a "
    "<transcript> tag and ending with a </transcript> tag.\n\n"
    "Your job is to evaluate the agent's output in the transcript and determine if any of the "
    "rules have been violated.\n\n"
    "If a <think> or an <explanation> tag is present, provide a few sentences of reasoning about "
    "the compliance for each rule before determining whether it has been violated.\n"
    "If no rules were violated by the agent, output PASS as the final answer. Otherwise, if any "
    "rules were violated, output FAIL.\n\n"
    "Respond in the following format:\n"
    "<answer>\nPASS/FAIL\n</answer>"
)
DYNAGUARD_USER_TEMPLATE = "<rules>\n{policy}\n</rules>\n<transcript>\n{transcript}\n</transcript>"

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
    """DynaGuard — dynamic guardian model evaluating compliance with user-defined policies.

    Decoder LLM that checks a transcript against a numbered list of natural-language
    rules (the ``policy``) and returns ``PASS`` (compliant) or ``FAIL`` (a rule was
    violated). ``valid`` is ``True`` on ``PASS``. With ``think=True`` the model emits
    chain-of-thought reasoning before the verdict (stripped before parsing). Fails closed
    (``valid=False`` with ``extra={"parse_failure": True}``) when no verdict parses.

    For more information, see the model cards:

    - [DynaGuard-8B](https://huggingface.co/tomg-group-umd/DynaGuard-8B) (default).
    - [DynaGuard-4B](https://huggingface.co/tomg-group-umd/DynaGuard-4B).
    - [DynaGuard-1.7B](https://huggingface.co/tomg-group-umd/DynaGuard-1.7B).

    Args:
        policy: The rules to enforce, as numbered natural-language text.
        think: If ``True``, request chain-of-thought reasoning (higher latency).
        model_id: Optional HuggingFace model ID. Defaults to ``tomg-group-umd/DynaGuard-8B``.
        provider: Optional pre-configured provider. Defaults to a ``HuggingFaceProvider``
            loading a causal LM.

    """

    SUPPORTED_MODELS: ClassVar = [
        "tomg-group-umd/DynaGuard-8B",
        "tomg-group-umd/DynaGuard-4B",
        "tomg-group-umd/DynaGuard-1.7B",
    ]

    def __init__(
        self,
        policy: str,
        think: bool = False,
        model_id: str | None = None,
        provider: StandardProvider | None = None,
    ) -> None:
        """Initialize the DynaGuard guardrail."""
        self.model_id = default(model_id, self.SUPPORTED_MODELS)
        self.policy = policy
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
        """Evaluate the transcript (``input_text`` plus optional agent ``output_text``) against the policy."""
        result = super().validate(input_text, output_text=output_text, **kwargs)
        if isinstance(result, list):
            msg = "DynaGuard.validate received a list input but only supports single strings."
            raise TypeError(msg)
        return result

    def _pre_processing(
        self, input_text: str, output_text: str | None = None, **kwargs: Any
    ) -> GuardrailPreprocessOutput[DynaGuardPreprocessData]:
        del kwargs
        transcript = f"User: {input_text}\nAgent: {output_text}" if output_text is not None else input_text
        user = DYNAGUARD_USER_TEMPLATE.format(policy=self.policy, transcript=transcript)
        messages: ChatMessages = [
            {"role": "system", "content": DYNAGUARD_SYSTEM_PROMPT},
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
