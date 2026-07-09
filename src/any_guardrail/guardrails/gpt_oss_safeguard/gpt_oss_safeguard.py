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

GptOssSafeguardPreprocessData = AnyDict
GptOssSafeguardInferenceData = AnyDict

# Appended to the user-supplied policy so the (reasoning) model ends with a parseable verdict.
OUTPUT_INSTRUCTION = (
    "After your analysis, end your reply with a final line containing exactly one word: "
    "VIOLATION if the content violates the policy, or SAFE if it does not."
)

MAX_NEW_TOKENS = 512
_VERDICT_PATTERN = re.compile(r"\b(VIOLATION|SAFE)\b", re.IGNORECASE)


class GptOssSafeguard(ThreeStageGuardrail[GptOssSafeguardPreprocessData, GptOssSafeguardInferenceData]):
    """gpt-oss-safeguard — policy-grounded reasoning safety classifier that judges text against a bring-your-own written policy (OpenAI).

    A reasoning LLM that classifies content against a written ``policy`` supplied at
    construction (bring-your-own-taxonomy). The policy becomes the system message; the
    model reasons (OpenAI harmony format) and emits a verdict. A short output instruction
    is appended so the reply ends with ``VIOLATION`` or ``SAFE``.

    Verdict mapping onto ``GuardrailOutput``:

    - ``valid`` is ``True`` on ``SAFE`` and ``False`` on ``VIOLATION``.
    - ``categories`` holds a single ``policy_violation`` entry (``triggered=True``
      on ``VIOLATION``).
    - ``score`` is not populated — the model emits a discrete verdict, not a
      calibrated probability.
    - ``extra["verdict"]`` carries the raw verdict string and ``explanation`` the
      full generation, including the model's reasoning.
    - Fails closed (``valid=False`` with ``extra={"parse_failure": True}``) when no
      verdict parses.

    Input is a single string ``input_text`` (the content to moderate, sent as the
    user message); list/batch input is not supported, and there is no separate
    response argument — include any conversation context in the text itself.

    Note: the 120B variant is large; ``gpt-oss-safeguard-20b`` is the practical default.

    For more information, see the model cards:

    - [gpt-oss-safeguard-20b](https://huggingface.co/openai/gpt-oss-safeguard-20b) (default).
    - [gpt-oss-safeguard-120b](https://huggingface.co/openai/gpt-oss-safeguard-120b).

    Args:
        policy: The written safety policy the model evaluates content against.
        model_id: Optional HuggingFace model ID. Defaults to ``openai/gpt-oss-safeguard-20b``.
        provider: Optional pre-configured provider. Defaults to a ``HuggingFaceProvider``
            loading a causal LM.

    """

    SUPPORTED_MODELS: ClassVar = [
        "openai/gpt-oss-safeguard-20b",
        "openai/gpt-oss-safeguard-120b",
    ]

    def __init__(
        self,
        policy: str,
        model_id: str | None = None,
        provider: StandardProvider | None = None,
    ) -> None:
        """Initialize the gpt-oss-safeguard guardrail.

        Args:
            policy: The written safety policy (bring-your-own-taxonomy) the model
                evaluates content against — e.g. a numbered list of disallowed-content
                rules with definitions and examples. It is installed as the system
                message, with a short output instruction appended so the model ends
                its reply with ``VIOLATION`` or ``SAFE``.
            model_id: Optional HuggingFace model ID. Must be one of ``SUPPORTED_MODELS``:
                ``openai/gpt-oss-safeguard-20b`` (default) or
                ``openai/gpt-oss-safeguard-120b``.
            provider: Optional pre-configured provider (e.g. a ``LlamafileProvider`` or
                a customized ``HuggingFaceProvider``). Defaults to a
                ``HuggingFaceProvider`` loading the model with
                ``AutoModelForCausalLM``/``AutoTokenizer``; when a
                ``HuggingFaceProvider`` is supplied, the causal-LM loader classes are
                enforced at load time.

        Raises:
            ValueError: If ``model_id`` is not in ``SUPPORTED_MODELS``.

        """
        self.model_id = default(model_id, self.SUPPORTED_MODELS)
        self.policy = policy
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

    def validate(self, input_text: str, **kwargs: Any) -> GuardrailOutput:  # type: ignore[override]
        """Classify ``input_text`` against the configured policy.

        Args:
            input_text: The content to moderate, sent as the user message beneath the
                policy system message. Single string only; list input raises
                ``TypeError``.
            **kwargs: Accepted for pipeline-signature compatibility; ignored by this
                guardrail's preprocessing.

        Returns:
            GuardrailOutput where ``valid=True`` means the model answered ``SAFE``,
            ``categories`` holds one ``policy_violation`` entry, ``extra["verdict"]``
            is the raw ``SAFE``/``VIOLATION`` string, and ``explanation`` is the full
            generation (reasoning included). Fails closed (``valid=False`` with
            ``extra={"parse_failure": True}``) when no verdict parses.

        Raises:
            TypeError: If ``input_text`` is a list.

        """
        result = super().validate(input_text, **kwargs)
        if isinstance(result, list):
            msg = "GptOssSafeguard.validate received a list input but only supports single strings."
            raise TypeError(msg)
        return result

    def _pre_processing(
        self, input_text: str, **kwargs: Any
    ) -> GuardrailPreprocessOutput[GptOssSafeguardPreprocessData]:
        del kwargs
        messages: ChatMessages = [
            {"role": "system", "content": f"{self.policy}\n\n{OUTPUT_INSTRUCTION}"},
            {"role": "user", "content": input_text},
        ]
        return GuardrailPreprocessOutput(data={"messages": messages})

    def _inference(
        self, model_inputs: GuardrailPreprocessOutput[GptOssSafeguardPreprocessData]
    ) -> GuardrailInferenceOutput[GptOssSafeguardInferenceData]:
        return self.provider.generate_chat(
            messages=model_inputs.data["messages"], max_new_tokens=MAX_NEW_TOKENS, do_sample=False
        )

    def _post_processing(
        self, model_outputs: GuardrailInferenceOutput[GptOssSafeguardInferenceData]
    ) -> GuardrailOutput:
        text = model_outputs.data["generated_text"]
        matches = _VERDICT_PATTERN.findall(text)
        if not matches:
            return GuardrailOutput(valid=False, explanation=text, extra={"parse_failure": True})
        verdict = matches[-1].upper()
        violation = verdict == "VIOLATION"
        return GuardrailOutput(
            valid=not violation,
            explanation=text,
            categories=[CategoryResult(name="policy_violation", triggered=violation)],
            extra={"verdict": verdict},
            usage=GuardrailUsage(
                prompt_tokens=model_outputs.data.get("prompt_token_count"),
                completion_tokens=model_outputs.data.get("completion_token_count"),
            ),
        )
