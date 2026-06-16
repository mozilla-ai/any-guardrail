import re
from typing import Any, ClassVar

from any_guardrail.base import GuardrailOutput, ThreeStageGuardrail
from any_guardrail.guardrails.utils import default, normalize_rubric_to_risk
from any_guardrail.providers.base import StandardProvider
from any_guardrail.providers.huggingface import HuggingFaceProvider
from any_guardrail.types import (
    AnyDict,
    ChatMessages,
    GuardrailInferenceOutput,
    GuardrailPreprocessOutput,
    GuardrailUsage,
)

CompassJudgerPreprocessData = AnyDict
CompassJudgerInferenceData = AnyDict

# CompassJudger is a generic prompt-driven judge with no canonical pointwise format, so we
# define one: score on a fixed 1-10 scale and emit it as ``Rating: [[X]]`` for robust parsing.
SCORE_MIN = 1
SCORE_MAX = 10

POINTWISE_PROMPT = """You are an impartial judge. Rate the response below against the criteria and rubric on an integer scale from 1 to 10.

Criteria:
{criteria}

Rubric:
{rubric}

Instruction:
{instruction}

Response:
{response}

First give a brief justification, then end your reply with the rating in the exact format: Rating: [[X]] where X is an integer from 1 to 10."""

MAX_NEW_TOKENS = 1024
_RATING_PATTERN = re.compile(r"\[\[\s*(\d{1,2})\s*\]\]")


class CompassJudger(ThreeStageGuardrail[CompassJudgerPreprocessData, CompassJudgerInferenceData]):
    """CompassJudger (OpenCompass) — generalist LLM judge.

    Scores a response against a user-defined ``criteria`` and ``rubric`` on a 1-10 scale.
    CompassJudger has no canonical pointwise output format, so this guardrail instructs it
    to emit ``Rating: [[X]]``. ``valid`` maps the rating through ``pass_threshold``; ``score``
    is normalized onto the canonical risk axis; ``explanation`` is the model's justification.
    Fails closed (``valid=False`` with ``extra={"parse_failure": True}``) when no rating parses.

    For more information, see the model cards:

    - [CompassJudger-2-7B-Instruct](https://huggingface.co/opencompass/CompassJudger-2-7B-Instruct) (default).
    - [CompassJudger-2-32B-Instruct](https://huggingface.co/opencompass/CompassJudger-2-32B-Instruct).
    - [CompassJudger-1-1.5B-Instruct](https://huggingface.co/opencompass/CompassJudger-1-1.5B-Instruct).
    - [CompassJudger-1-7B-Instruct](https://huggingface.co/opencompass/CompassJudger-1-7B-Instruct).
    - [CompassJudger-1-14B-Instruct](https://huggingface.co/opencompass/CompassJudger-1-14B-Instruct).
    - [CompassJudger-1-32B-Instruct](https://huggingface.co/opencompass/CompassJudger-1-32B-Instruct).

    Args:
        criteria: A description of what is being judged.
        rubric: The scoring rubric guidance.
        pass_threshold: The rating (1-10) at or above which the response passes.
        higher_is_better: Whether higher ratings mean better. Defaults to ``True``.
        model_id: Optional HuggingFace model ID. Defaults to ``opencompass/CompassJudger-2-7B-Instruct``.
        provider: Optional pre-configured provider. Defaults to a ``HuggingFaceProvider``
            loading a causal LM.

    """

    SUPPORTED_MODELS: ClassVar = [
        "opencompass/CompassJudger-2-7B-Instruct",
        "opencompass/CompassJudger-2-32B-Instruct",
        "opencompass/CompassJudger-1-1.5B-Instruct",
        "opencompass/CompassJudger-1-7B-Instruct",
        "opencompass/CompassJudger-1-14B-Instruct",
        "opencompass/CompassJudger-1-32B-Instruct",
    ]

    def __init__(
        self,
        criteria: str,
        rubric: str,
        pass_threshold: int,
        higher_is_better: bool = True,
        model_id: str | None = None,
        provider: StandardProvider | None = None,
    ) -> None:
        """Initialize the CompassJudger guardrail."""
        self.model_id = default(model_id, self.SUPPORTED_MODELS)
        self.criteria = criteria
        self.rubric = rubric
        self.pass_threshold = pass_threshold
        self.higher_is_better = higher_is_better
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
        """Judge ``output_text`` (the response) given ``input_text`` (the instruction)."""
        result = super().validate(input_text, output_text=output_text, **kwargs)
        if isinstance(result, list):
            msg = "CompassJudger.validate received a list input but only supports single strings."
            raise TypeError(msg)
        return result

    def _pre_processing(
        self, input_text: str, output_text: str | None = None, **kwargs: Any
    ) -> GuardrailPreprocessOutput[CompassJudgerPreprocessData]:
        del kwargs
        prompt = POINTWISE_PROMPT.format(
            criteria=self.criteria,
            rubric=self.rubric,
            instruction=input_text,
            response=output_text if output_text is not None else input_text,
        )
        messages: ChatMessages = [{"role": "user", "content": prompt}]
        return GuardrailPreprocessOutput(data={"messages": messages})

    def _inference(
        self, model_inputs: GuardrailPreprocessOutput[CompassJudgerPreprocessData]
    ) -> GuardrailInferenceOutput[CompassJudgerInferenceData]:
        return self.provider.generate_chat(
            messages=model_inputs.data["messages"], max_new_tokens=MAX_NEW_TOKENS, do_sample=False
        )

    def _post_processing(self, model_outputs: GuardrailInferenceOutput[CompassJudgerInferenceData]) -> GuardrailOutput:
        text = model_outputs.data["generated_text"]
        # The verdict is the LAST ``[[X]]``: the justification may quote other bracketed
        # numbers before the final rating, so leftmost-match would be wrong. Take the final one.
        matches = list(_RATING_PATTERN.finditer(text))
        if not matches:
            return GuardrailOutput(valid=False, explanation=text, extra={"parse_failure": True})
        rating = int(matches[-1].group(1))
        if not SCORE_MIN <= rating <= SCORE_MAX:
            return GuardrailOutput(valid=False, explanation=text, extra={"parse_failure": True})
        passed = rating >= self.pass_threshold if self.higher_is_better else rating <= self.pass_threshold
        return GuardrailOutput(
            valid=passed,
            score=normalize_rubric_to_risk(rating, SCORE_MIN, SCORE_MAX, higher_is_better=self.higher_is_better),
            explanation=text,
            extra={"rubric_score": rating},
            usage=GuardrailUsage(
                prompt_tokens=model_outputs.data.get("prompt_token_count"),
                completion_tokens=model_outputs.data.get("completion_token_count"),
            ),
        )
