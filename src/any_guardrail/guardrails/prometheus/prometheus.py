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

PrometheusPreprocessData = AnyDict
PrometheusInferenceData = AnyDict

ABS_SYSTEM_PROMPT = (
    "You are a fair judge assistant tasked with providing clear, objective feedback based on specific "
    "criteria, ensuring each assessment reflects the absolute standards set for performance."
)

ABSOLUTE_PROMPT = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Reference Answer (Score 5):
{reference_answer}

###Score Rubrics:
{rubric}

###Feedback: """

SCORE_MIN = 1
SCORE_MAX = 5
MAX_NEW_TOKENS = 1024

_RESULT_PATTERN = re.compile(r"(?:\[RESULT\]|\[SCORE\]|Result:|Score:)\s*\(?\[?\s*([1-5])\b", re.IGNORECASE)


class Prometheus(ThreeStageGuardrail[PrometheusPreprocessData, PrometheusInferenceData]):
    """Prometheus — open rubric-based LLM judge (KAIST).

    Evaluates a response against a user-defined ``rubric`` on a 1-5 scale (absolute
    grading) and returns feedback plus an integer score. ``valid`` maps the score
    through ``pass_threshold``; ``score`` is the rubric normalized onto the canonical
    risk axis; ``explanation`` is the model's feedback. Fails closed (``valid=False``
    with ``extra={"parse_failure": True}``) when no score parses.

    For more information, see the model cards:

    - [prometheus-7b-v2.0](https://huggingface.co/prometheus-eval/prometheus-7b-v2.0) (default).
    - [prometheus-8x7b-v2.0](https://huggingface.co/prometheus-eval/prometheus-8x7b-v2.0).
    - [prometheus-7b-v1.0](https://huggingface.co/prometheus-eval/prometheus-7b-v1.0).
    - [prometheus-13b-v1.0](https://huggingface.co/prometheus-eval/prometheus-13b-v1.0).

    Args:
        rubric: The score rubric (criteria plus ``Score 1:`` … ``Score 5:`` descriptions).
        pass_threshold: The score (1-5) at or above which the response passes.
        reference_answer: Optional reference answer that would score 5.
        higher_is_better: Whether higher scores mean better. Defaults to ``True``.
        model_id: Optional HuggingFace model ID. Defaults to ``prometheus-eval/prometheus-7b-v2.0``.
        provider: Optional pre-configured provider. Defaults to a ``HuggingFaceProvider``
            loading a causal LM.

    """

    SUPPORTED_MODELS: ClassVar = [
        "prometheus-eval/prometheus-7b-v2.0",
        "prometheus-eval/prometheus-8x7b-v2.0",
        "prometheus-eval/prometheus-7b-v1.0",
        "prometheus-eval/prometheus-13b-v1.0",
    ]

    def __init__(
        self,
        rubric: str,
        pass_threshold: int,
        reference_answer: str | None = None,
        higher_is_better: bool = True,
        model_id: str | None = None,
        provider: StandardProvider | None = None,
    ) -> None:
        """Initialize the Prometheus guardrail."""
        self.model_id = default(model_id, self.SUPPORTED_MODELS)
        self.rubric = rubric
        self.pass_threshold = pass_threshold
        self.reference_answer = reference_answer
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
            msg = "Prometheus.validate received a list input but only supports single strings."
            raise TypeError(msg)
        return result

    def _pre_processing(
        self, input_text: str, output_text: str | None = None, **kwargs: Any
    ) -> GuardrailPreprocessOutput[PrometheusPreprocessData]:
        del kwargs
        user = ABSOLUTE_PROMPT.format(
            instruction=input_text,
            response=output_text if output_text is not None else input_text,
            reference_answer=self.reference_answer or "",
            rubric=self.rubric,
        )
        messages: ChatMessages = [
            {"role": "system", "content": ABS_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
        return GuardrailPreprocessOutput(data={"messages": messages})

    def _inference(
        self, model_inputs: GuardrailPreprocessOutput[PrometheusPreprocessData]
    ) -> GuardrailInferenceOutput[PrometheusInferenceData]:
        return self.provider.generate_chat(
            messages=model_inputs.data["messages"], max_new_tokens=MAX_NEW_TOKENS, do_sample=False
        )

    def _post_processing(self, model_outputs: GuardrailInferenceOutput[PrometheusInferenceData]) -> GuardrailOutput:
        text = model_outputs.data["generated_text"]
        # The verdict is the LAST score marker: feedback often references other rubric
        # levels inline (e.g. "a Score: 2 response would... [RESULT] 4"), so leftmost-match
        # would capture the wrong number. Take the final match.
        matches = list(_RESULT_PATTERN.finditer(text))
        if not matches:
            return GuardrailOutput(valid=False, explanation=text, extra={"parse_failure": True})
        rubric_score = int(matches[-1].group(1))
        passed = rubric_score >= self.pass_threshold if self.higher_is_better else rubric_score <= self.pass_threshold
        feedback = text.split("[RESULT]")[0].replace("Feedback:", "").strip() or text
        return GuardrailOutput(
            valid=passed,
            score=normalize_rubric_to_risk(rubric_score, SCORE_MIN, SCORE_MAX, higher_is_better=self.higher_is_better),
            explanation=feedback,
            extra={"rubric_score": rubric_score},
            usage=GuardrailUsage(
                prompt_tokens=model_outputs.data.get("prompt_token_count"),
                completion_tokens=model_outputs.data.get("completion_token_count"),
            ),
        )
