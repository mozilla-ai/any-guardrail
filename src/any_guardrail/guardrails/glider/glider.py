import re
from typing import ClassVar

from transformers import pipeline

from any_guardrail.base import GuardrailName, GuardrailOutput, ThreeStageGuardrail
from any_guardrail.guardrails.utils import default, normalize_rubric_to_risk
from any_guardrail.prompt_registry import PROMPT_REGISTRY, resolve_prompt
from any_guardrail.prompts import PromptSpec, PromptTemplate
from any_guardrail.providers.base import StandardProvider
from any_guardrail.registry import GUARDRAIL_METADATA
from any_guardrail.taxonomy import GuardrailMetadata
from any_guardrail.types import ChatMessages, GuardrailInferenceOutput, GuardrailPreprocessOutput

SCORE_PATTERN = re.compile(r"<score>\s*(\d+)\s*</score>")
REASONING_PATTERN = re.compile(r"<reasoning>\s*(.*?)\s*</reasoning>", re.DOTALL)
HIGHLIGHT_PATTERN = re.compile(r"<highlight>\s*(.*?)\s*</highlight>", re.DOTALL)

SYSTEM_PROMPT_GLIDER = PROMPT_REGISTRY[GuardrailName.GLIDER].resolve().segments["system"]
"""GLIDER evaluation prompt (registry-sourced); fills ``{data}`` / ``{pass_criteria}`` / ``{rubric}``."""

INPUT_OUTPUT_DATA_FORMAT = PROMPT_REGISTRY[GuardrailName.GLIDER].resolve().segments["input_output"]
"""GLIDER input+output data wrapper (registry-sourced); fills ``{input_text}`` / ``{output_text}``."""

INPUT_DATA_FORMAT = PROMPT_REGISTRY[GuardrailName.GLIDER].resolve().segments["input"]
"""GLIDER input-only data wrapper (registry-sourced); fills ``{input_text}``."""


class Glider(ThreeStageGuardrail[ChatMessages, str]):
    """GLIDER — prompt-based LLM judge that grades text against user-supplied pass criteria and rubric, returning reasoning and highlighted phrases (Patronus AI).

    GLIDER is a compact (3B) evaluator LLM fine-tuned to score arbitrary text on
    arbitrary user-defined criteria. Each call wraps the text in GLIDER's evaluation
    prompt together with ``pass_criteria`` and ``rubric``; the model replies with a
    ``<reasoning>`` block, a ``<highlight>`` list of the decisive words/phrases, and
    an integer ``<score>``.

    Verdict mapping onto ``GuardrailOutput``:

    - ``valid`` is ``rubric_score >= pass_threshold`` (or ``<=`` when
      ``higher_is_better=False``).
    - ``score`` (canonical risk: higher = riskier) is populated only when
      ``score_range`` is supplied — the rubric score is normalized onto [0, 1] and
      inverted when higher rubric values mean better text. Otherwise ``score`` is
      ``None``.
    - ``explanation`` is the model's ``<reasoning>`` block (the full generation when
      the block is missing).
    - ``extra`` holds the raw integer ``rubric_score`` and the ``highlights`` string.
    - When no ``<score>`` can be parsed, the output fails closed: ``valid=False``
      with ``extra={"parse_failure": True}``.

    Inputs are single strings: ``input_text`` (required) plus an optional
    ``output_text`` judged alongside it (typically a model response); list/batch
    input is not supported.

    Note: this guardrail runs the model through a ``transformers`` text-generation
    pipeline directly; the ``provider`` argument is reserved for future
    extensibility and is currently unused.

    For more information, see:

    - [GLIDER model card](https://huggingface.co/PatronusAI/glider)
    - [GLIDER: Grading LLM Interactions and Decisions using Explainable Ranking](https://arxiv.org/abs/2412.14140)

    Args:
        pass_criteria: A question or description of what you are validating.
        rubric: A scoring rubric, describing to the model how to score the provided data.
        pass_threshold: The rubric score at which the text counts as passing. ``valid`` is
            ``rubric_score >= pass_threshold`` (or ``<=`` when ``higher_is_better`` is False).
        model_id: HuggingFace path to model. Defaults to ``PatronusAI/glider``.
        provider: Reserved for future extensibility. Currently unused.
        higher_is_better: Whether higher rubric scores mean better/passing text. Set to
            False for rubrics where higher scores mean worse text.
        score_range: Optional ``(min, max)`` bounds of the rubric scale. GLIDER's rubric is
            free text, so the bounds can't be inferred; supply them to get a normalized
            canonical risk in ``score``. When omitted, ``score`` is None and the raw rubric
            value is still available in ``extra["rubric_score"]``.

    Raises:
        ValueError: Can only use model path to GLIDER from HuggingFace.

    """

    SUPPORTED_MODELS: ClassVar = ["PatronusAI/glider"]

    METADATA: ClassVar[GuardrailMetadata] = GUARDRAIL_METADATA[GuardrailName.GLIDER]

    PROMPT: ClassVar[PromptSpec] = PROMPT_REGISTRY[GuardrailName.GLIDER]

    def __init__(
        self,
        pass_criteria: str,
        rubric: str,
        pass_threshold: int,
        model_id: str | None = None,
        provider: StandardProvider | None = None,  # Reserved for future extensibility
        higher_is_better: bool = True,
        score_range: tuple[int, int] | None = None,
        prompt: PromptTemplate | None = None,
        prompt_version: str | None = None,
    ) -> None:
        """Initialize the GLIDER guardrail.

        Args:
            pass_criteria: A question or description of what is being judged, e.g.
                ``"Is the response free of unsupported medical claims?"``.
            rubric: A free-text scoring rubric telling the model what each score means,
                e.g. ``"0: contains unsupported claims. 1: all claims are supported."``.
            pass_threshold: The rubric score at which the text counts as passing.
                ``valid`` is ``rubric_score >= pass_threshold`` (or ``<=`` when
                ``higher_is_better=False``). Must be on the same scale as the rubric.
            model_id: Optional HuggingFace model ID. Must be one of ``SUPPORTED_MODELS``;
                defaults to ``PatronusAI/glider``.
            provider: Reserved for future extensibility; currently unused. GLIDER runs
                through a ``transformers`` text-generation pipeline instead.
            higher_is_better: Whether higher rubric scores mean better/passing text.
                Set to ``False`` for rubrics where higher scores mean worse text (e.g. a
                severity scale).
            score_range: Optional ``(min, max)`` bounds of the rubric scale, e.g.
                ``(0, 1)`` or ``(1, 5)``. Supplying it enables the normalized canonical
                risk in ``GuardrailOutput.score``; when omitted, ``score`` is ``None``
                and the raw rubric value is still available in ``extra["rubric_score"]``.
            prompt: Optional prompt-template override, used as-is (system prompt filling ``{data}`` /
                ``{pass_criteria}`` / ``{rubric}`` plus the ``input`` / ``input_output`` data
                wrappers). Defaults to ``None`` — the registry default, or the version named by
                ``prompt_version``.
            prompt_version: Registered prompt version to use when ``prompt`` is not given. Defaults
                to ``None`` (the default version). See ``AnyGuardrail.list_prompt_versions``.

        Raises:
            ValueError: If ``model_id`` is not in ``SUPPORTED_MODELS``.

        """
        self.model_id = default(model_id, self.SUPPORTED_MODELS)
        self.pass_criteria = pass_criteria
        self.rubric = rubric
        self.pass_threshold = pass_threshold
        self.higher_is_better = higher_is_better
        self.score_range = score_range
        self._prompt = resolve_prompt(GuardrailName.GLIDER, prompt, prompt_version)
        self.provider = provider  # Reserved for future extensibility
        self.model = pipeline("text-generation", self.model_id, max_new_tokens=2048, return_full_text=False)

    def validate(self, input_text: str, output_text: str | None = None) -> GuardrailOutput:  # type: ignore[override]
        """Use the provided pass criteria and rubric to judge the input and output text provided.

        Args:
            input_text: The text to evaluate, wrapped in ``<INPUT>`` tags in GLIDER's
                evaluation prompt. Typically the user prompt or the standalone text
                being judged. Single string only; list/batch input is not supported.
            output_text: Optional second text, wrapped in ``<OUTPUT>`` tags and judged
                alongside ``input_text`` — typically the model response when the pass
                criteria compare a response against a prompt (e.g.
                ``"Does the OUTPUT answer the question in the INPUT?"``).

        Returns:
            GuardrailOutput where ``valid`` maps the rubric score through
            ``pass_threshold``, ``score`` is the normalized canonical risk when
            ``score_range`` was supplied (otherwise ``None``), ``explanation`` is the
            model's reasoning, and ``extra`` holds ``rubric_score`` and
            ``highlights``. When the rubric score cannot be parsed, the output fails
            closed (``valid=False`` with ``extra={"parse_failure": True}``).

        """
        return self._execute(input_text, output_text)

    def _pre_processing(
        self, input_text: str, output_text: str | None = None
    ) -> GuardrailPreprocessOutput[ChatMessages]:
        """Format GLIDER's evaluation prompt as a single-turn chat message.

        Args:
            input_text: Text placed inside the ``<INPUT>`` block of the prompt.
            output_text: Optional text placed inside the ``<OUTPUT>`` block; when
                ``None``, the prompt contains only the ``<INPUT>`` block.

        Returns:
            GuardrailPreprocessOutput wrapping the one-message chat prompt.

        """
        if output_text is None:
            data = self._prompt.segments["input"].format(input_text=input_text)
        else:
            data = self._prompt.segments["input_output"].format(input_text=input_text, output_text=output_text)
        prompt = self._prompt.segments["system"].format(data=data, pass_criteria=self.pass_criteria, rubric=self.rubric)
        return GuardrailPreprocessOutput(data=[{"role": "user", "content": prompt}])

    def _inference(self, message: GuardrailPreprocessOutput[ChatMessages]) -> GuardrailInferenceOutput[str]:
        """Run text-generation pipeline on chat messages."""
        generated_text: str = self.model(message.data)[0]["generated_text"]  # type: ignore[assignment]
        return GuardrailInferenceOutput(data=generated_text)

    def _post_processing(self, model_outputs: GuardrailInferenceOutput[str]) -> GuardrailOutput:
        generated_text = model_outputs.data
        score_match = SCORE_PATTERN.search(generated_text)
        if score_match is None:
            return GuardrailOutput(valid=False, explanation=generated_text, extra={"parse_failure": True})

        rubric_score = int(score_match.group(1))
        passed = rubric_score >= self.pass_threshold if self.higher_is_better else rubric_score <= self.pass_threshold
        reasoning_match = REASONING_PATTERN.search(generated_text)
        highlight_match = HIGHLIGHT_PATTERN.search(generated_text)
        # GLIDER's rubric is free text, so a canonical risk score is only
        # available when the caller supplies the scale via ``score_range``.
        score = (
            normalize_rubric_to_risk(
                rubric_score, self.score_range[0], self.score_range[1], higher_is_better=self.higher_is_better
            )
            if self.score_range is not None
            else None
        )
        return GuardrailOutput(
            valid=passed,
            score=score,
            explanation=reasoning_match.group(1) if reasoning_match else generated_text,
            extra={
                "rubric_score": rubric_score,
                "highlights": highlight_match.group(1) if highlight_match else None,
            },
        )
