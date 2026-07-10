import re
from typing import Any, ClassVar

from any_guardrail.base import GuardrailName, GuardrailOutput, ThreeStageGuardrail
from any_guardrail.guardrails.utils import default, normalize_rubric_to_risk
from any_guardrail.prompt_registry import PROMPT_REGISTRY, resolve_prompt
from any_guardrail.prompts import PromptSpec, PromptTemplate
from any_guardrail.providers.base import StandardProvider
from any_guardrail.providers.huggingface import HuggingFaceProvider
from any_guardrail.registry import GUARDRAIL_METADATA
from any_guardrail.taxonomy import GuardrailMetadata
from any_guardrail.types import (
    AnyDict,
    ChatMessages,
    GuardrailInferenceOutput,
    GuardrailPreprocessOutput,
    GuardrailUsage,
)

PrometheusPreprocessData = AnyDict
PrometheusInferenceData = AnyDict

ABS_SYSTEM_PROMPT = PROMPT_REGISTRY[GuardrailName.PROMETHEUS].resolve().segments["system"]
"""Prometheus absolute-grading system prompt (registry-sourced)."""

ABSOLUTE_PROMPT = PROMPT_REGISTRY[GuardrailName.PROMETHEUS].resolve().segments["user"]
"""Prometheus absolute-grading user template (registry-sourced); fills instruction/response/reference_answer/rubric."""

SCORE_MIN = 1
SCORE_MAX = 5
MAX_NEW_TOKENS = 1024

_RESULT_PATTERN = re.compile(r"(?:\[RESULT\]|\[SCORE\]|Result:|Score:)\s*\(?\[?\s*([1-5])\b", re.IGNORECASE)


class Prometheus(ThreeStageGuardrail[PrometheusPreprocessData, PrometheusInferenceData]):
    """Prometheus — open rubric-based LLM judge grading a response on a user-defined 1-5 rubric (KAIST / prometheus-eval).

    Prometheus is an open-source decoder LLM specialized in evaluating other models'
    outputs. This guardrail drives it in **absolute grading** mode: each call wraps the
    instruction and response in Prometheus's evaluation prompt together with the caller's
    ``rubric`` (and an optional ``reference_answer`` that would score 5), and the model
    replies with written feedback followed by ``[RESULT] <n>`` where ``n`` is an integer
    1-5. It runs through ``provider.generate_chat``, so it can be served from either a
    ``HuggingFaceProvider`` or a ``LlamafileProvider``.

    Verdict mapping onto ``GuardrailOutput``:

    - ``valid`` is ``rubric_score >= pass_threshold`` (or ``<=`` when
      ``higher_is_better=False``).
    - ``score`` (canonical risk: higher = riskier) is the 1-5 rubric score normalized onto
      [0, 1] via ``normalize_rubric_to_risk`` — inverted when higher rubric values mean
      better, so a high-quality response yields a low risk.
    - ``explanation`` is the model's feedback (the text before the ``[RESULT]`` marker).
    - ``extra["rubric_score"]`` is the raw integer 1-5.
    - When no score can be parsed, the output fails closed: ``valid=False`` with
      ``extra={"parse_failure": True}``. The parser takes the **last** ``[RESULT]`` marker,
      because feedback often quotes other rubric levels inline.

    Inputs are single strings: ``input_text`` is the instruction and ``output_text`` is the
    response being graded (when ``output_text`` is omitted, ``input_text`` is graded as the
    response). List/batch input is not supported.

    For more information, see:

    - [prometheus-7b-v2.0 model card](https://huggingface.co/prometheus-eval/prometheus-7b-v2.0) (default)
    - [prometheus-8x7b-v2.0 model card](https://huggingface.co/prometheus-eval/prometheus-8x7b-v2.0)
    - [prometheus-7b-v1.0 model card](https://huggingface.co/prometheus-eval/prometheus-7b-v1.0)
    - [prometheus-13b-v1.0 model card](https://huggingface.co/prometheus-eval/prometheus-13b-v1.0)
    - [Prometheus 2: An Open Source Language Model Specialized in Evaluating Other Language Models (arXiv:2405.01535)](https://arxiv.org/abs/2405.01535)
    - [prometheus-eval/prometheus-eval on GitHub](https://github.com/prometheus-eval/prometheus-eval)

    Args:
        rubric: The score rubric — the evaluation criteria plus ``Score 1:`` … ``Score 5:``
            descriptions of what each level means.
        pass_threshold: The score (1-5) at or above which the response passes (or at or below
            when ``higher_is_better=False``).
        reference_answer: Optional gold answer that would earn a score of 5, given to the
            model as an anchor for the top of the scale.
        higher_is_better: Whether higher rubric scores mean better responses. Defaults to
            ``True``.
        model_id: Optional HuggingFace model ID; must be one of ``SUPPORTED_MODELS``.
            Defaults to ``prometheus-eval/prometheus-7b-v2.0``.
        provider: Optional pre-configured provider. Defaults to a ``HuggingFaceProvider``
            loading a causal LM.

    """

    SUPPORTED_MODELS: ClassVar = [
        "prometheus-eval/prometheus-7b-v2.0",
        "prometheus-eval/prometheus-8x7b-v2.0",
        "prometheus-eval/prometheus-7b-v1.0",
        "prometheus-eval/prometheus-13b-v1.0",
    ]

    METADATA: ClassVar[GuardrailMetadata] = GUARDRAIL_METADATA[GuardrailName.PROMETHEUS]

    PROMPT: ClassVar[PromptSpec] = PROMPT_REGISTRY[GuardrailName.PROMETHEUS]

    def __init__(
        self,
        rubric: str,
        pass_threshold: int,
        reference_answer: str | None = None,
        higher_is_better: bool = True,
        model_id: str | None = None,
        provider: StandardProvider | None = None,
        prompt: PromptTemplate | None = None,
        prompt_version: str | None = None,
    ) -> None:
        """Initialize the Prometheus guardrail.

        Args:
            rubric: The score rubric applied to every ``validate`` call — the evaluation
                criteria plus ``Score 1:`` … ``Score 5:`` descriptions, e.g.
                ``"Is the answer factually correct? Score 1: entirely wrong. ... Score 5: fully correct."``.
            pass_threshold: The score (1-5) at or above which the response passes (or at or
                below when ``higher_is_better=False``), e.g. ``4``.
            reference_answer: Optional gold answer that would earn a score of 5, supplied to
                the model as an anchor for the top of the scale. Defaults to ``None`` (no
                reference; an empty string is sent).
            higher_is_better: Whether higher rubric scores mean better responses. Set
                ``False`` for rubrics where a higher number is worse (e.g. a severity scale).
                Defaults to ``True``.
            model_id: Optional HuggingFace model ID; must be one of ``SUPPORTED_MODELS``.
                Defaults to ``prometheus-eval/prometheus-7b-v2.0``.
            provider: Optional pre-configured provider. When ``None``, a
                ``HuggingFaceProvider`` is built targeting ``AutoModelForCausalLM`` /
                ``AutoTokenizer`` (transformers is imported lazily here). Pass a
                ``LlamafileProvider`` to run a GGUF build without the huggingface extra.
            prompt: Optional prompt-template override, used as-is (must fill ``{instruction}`` /
                ``{response}`` / ``{reference_answer}`` / ``{rubric}``). Defaults to ``None`` — the
                registry default, or the version named by ``prompt_version``.
            prompt_version: Registered prompt version to use when ``prompt`` is not given. Defaults
                to ``None`` (the default version). See ``AnyGuardrail.list_prompt_versions``.

        Raises:
            ValueError: If ``model_id`` is not in ``SUPPORTED_MODELS``.

        """
        self.model_id = default(model_id, self.SUPPORTED_MODELS)
        self.rubric = rubric
        self.pass_threshold = pass_threshold
        self.reference_answer = reference_answer
        self.higher_is_better = higher_is_better
        self._prompt = resolve_prompt(GuardrailName.PROMETHEUS, prompt, prompt_version)
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
        """Judge ``output_text`` (the response) given ``input_text`` (the instruction).

        Args:
            input_text: The instruction the response was produced for, e.g.
                ``"Explain why the sky is blue to a five-year-old."``. Single string only;
                list/batch input is not supported and raises ``TypeError``.
            output_text: The response being graded against the rubric, e.g.
                ``"The sky is blue because sunlight scatters off the air."``. When ``None``,
                ``input_text`` itself is graded as the response.
            **kwargs: Ignored; accepted only so the signature matches the ``ThreeStageGuardrail``
                contract.

        Returns:
            GuardrailOutput where ``valid`` maps the rubric score through ``pass_threshold``,
            ``score`` is the 1-5 rubric score normalized onto the canonical risk axis
            (higher = riskier), ``explanation`` is the model's feedback, and
            ``extra["rubric_score"]`` is the raw integer. When no score can be parsed the
            output fails closed (``valid=False`` with ``extra={"parse_failure": True}``).

        """
        result = super().validate(input_text, output_text=output_text, **kwargs)
        if isinstance(result, list):
            msg = "Prometheus.validate received a list input but only supports single strings."
            raise TypeError(msg)
        return result

    def _pre_processing(
        self, input_text: str, output_text: str | None = None, **kwargs: Any
    ) -> GuardrailPreprocessOutput[PrometheusPreprocessData]:
        """Format Prometheus's absolute-grading prompt as a system + user chat message pair.

        Args:
            input_text: The instruction, substituted into the prompt's
                ``###The instruction to evaluate`` block.
            output_text: The response to grade, substituted into the ``###Response to
                evaluate`` block. When ``None``, ``input_text`` is reused as the response.
            **kwargs: Ignored (dropped via ``del kwargs``); present only for signature
                compatibility with the ``ThreeStageGuardrail`` contract.

        Returns:
            GuardrailPreprocessOutput wrapping the two-message chat prompt (system prompt
            plus the filled absolute-grading template) under the ``messages`` key.

        """
        del kwargs
        user = self._prompt.segments["user"].format(
            instruction=input_text,
            response=output_text if output_text is not None else input_text,
            reference_answer=self.reference_answer or "",
            rubric=self.rubric,
        )
        messages: ChatMessages = [
            {"role": "system", "content": self._prompt.segments["system"]},
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
