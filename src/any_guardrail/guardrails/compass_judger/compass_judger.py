import re
from typing import Any, ClassVar

from any_guardrail.base import GuardrailName, GuardrailOutput, ThreeStageGuardrail
from any_guardrail.guardrails.utils import broadcast_optional, default, normalize_rubric_to_risk
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

CompassJudgerPreprocessData = AnyDict
CompassJudgerInferenceData = AnyDict

# CompassJudger is a generic prompt-driven judge with no canonical pointwise format, so we
# define one: score on a fixed 1-10 scale and emit it as ``Rating: [[X]]`` for robust parsing.
SCORE_MIN = 1
SCORE_MAX = 10

POINTWISE_PROMPT = PROMPT_REGISTRY[GuardrailName.COMPASS_JUDGER].resolve().segments["user"]
"""Default CompassJudger pointwise 1-10 template (registry-sourced); fills criteria/rubric/instruction/response."""

MAX_NEW_TOKENS = 1024
_RATING_PATTERN = re.compile(r"\[\[\s*(\d{1,2})\s*\]\]")


class CompassJudger(ThreeStageGuardrail[CompassJudgerPreprocessData, CompassJudgerInferenceData]):
    """Generalist LLM judge that scores a response against user-defined criteria and rubric on a 1-10 scale.

    Decoder-LLM judge from the OpenCompass evaluation ecosystem. CompassJudger has no
    canonical pointwise output format, so this guardrail wraps the inputs in a fixed
    pointwise prompt instructing the model to give a brief justification and then emit
    its verdict as ``Rating: [[X]]`` with an integer from 1 to 10. The verdict is the
    *last* bracketed rating in the generation, so numbers the model quotes while
    justifying are not mistaken for the final rating.

    Inputs are strings: ``input_text`` is the instruction and ``output_text`` is the
    response being judged. When ``output_text`` is omitted, ``input_text`` itself is
    placed in the response slot and judged directly. ``input_text`` also accepts a
    ``list[str]`` to judge a batch in one real batched ``generate_chat`` call when the
    provider is a ``HuggingFaceProvider`` (``output_text`` may then be a matching-length
    list, a single value broadcast to every item, or omitted); other providers fall back
    to one call per item.

    Verdict mapping onto ``GuardrailOutput``:

    - ``valid`` is ``True`` when the rating passes ``pass_threshold``
      (``rating >= pass_threshold`` when ``higher_is_better``, ``<=`` otherwise).
    - ``score`` is the rating normalized onto the canonical risk axis in [0, 1]
      (higher = riskier), so a high rating under ``higher_is_better=True`` yields a
      low risk score.
    - ``explanation`` is the model's full generated justification.
    - ``extra["rubric_score"]`` is the raw 1-10 integer rating.
    - Fails closed (``valid=False`` with ``extra={"parse_failure": True}``) when no
      in-range rating parses.

    For more information, see:

    - [CompassJudger-2-7B-Instruct](https://huggingface.co/opencompass/CompassJudger-2-7B-Instruct) (default).
    - [CompassJudger-2-32B-Instruct](https://huggingface.co/opencompass/CompassJudger-2-32B-Instruct).
    - [CompassJudger-1-1.5B-Instruct](https://huggingface.co/opencompass/CompassJudger-1-1.5B-Instruct).
    - [CompassJudger-1-7B-Instruct](https://huggingface.co/opencompass/CompassJudger-1-7B-Instruct).
    - [CompassJudger-1-14B-Instruct](https://huggingface.co/opencompass/CompassJudger-1-14B-Instruct).
    - [CompassJudger-1-32B-Instruct](https://huggingface.co/opencompass/CompassJudger-1-32B-Instruct).
    - [CompassJudger GitHub repository](https://github.com/open-compass/CompassJudger).
    - [CompassJudger-2 paper (arXiv:2507.09104)](https://arxiv.org/abs/2507.09104).
    - [CompassJudger-1 paper (arXiv:2410.16256)](https://arxiv.org/abs/2410.16256).

    Args:
        criteria: A description of what is being judged, e.g.
            ``"Helpfulness of the response to the user's question"``.
        rubric: The scoring-rubric guidance describing what low vs. high ratings mean.
        pass_threshold: The 1-10 rating at which the response passes. With
            ``higher_is_better=True``, ratings at or above it yield ``valid=True``.
        higher_is_better: Whether higher ratings mean better text. Set to ``False``
            for rubrics where higher ratings mean worse text; the pass comparison and
            the risk normalization flip accordingly. Defaults to ``True``.
        model_id: Optional HuggingFace model ID from ``SUPPORTED_MODELS``. Defaults to
            ``opencompass/CompassJudger-2-7B-Instruct``.
        provider: Optional pre-configured provider (e.g. a ``LlamafileProvider`` or a
            customized ``HuggingFaceProvider``). Defaults to a ``HuggingFaceProvider``
            loading a causal LM with the portable SDPA attention kernel.

    """

    SUPPORTED_MODELS: ClassVar = [
        "opencompass/CompassJudger-2-7B-Instruct",
        "opencompass/CompassJudger-2-32B-Instruct",
        "opencompass/CompassJudger-1-1.5B-Instruct",
        "opencompass/CompassJudger-1-7B-Instruct",
        "opencompass/CompassJudger-1-14B-Instruct",
        "opencompass/CompassJudger-1-32B-Instruct",
    ]

    METADATA: ClassVar[GuardrailMetadata] = GUARDRAIL_METADATA[GuardrailName.COMPASS_JUDGER]

    PROMPT: ClassVar[PromptSpec] = PROMPT_REGISTRY[GuardrailName.COMPASS_JUDGER]

    def __init__(
        self,
        criteria: str,
        rubric: str,
        pass_threshold: int,
        higher_is_better: bool = True,
        model_id: str | None = None,
        provider: StandardProvider | None = None,
        prompt: PromptTemplate | None = None,
        prompt_version: str | None = None,
    ) -> None:
        """Initialize the CompassJudger guardrail.

        Args:
            criteria: A description of what is being judged, e.g.
                ``"Helpfulness of the response to the user's question"``.
            rubric: The scoring-rubric guidance describing what low vs. high ratings
                mean, e.g. ``"1-3: unhelpful ... 8-10: fully answers the question"``.
            pass_threshold: The 1-10 rating at which the response passes. With
                ``higher_is_better=True``, ratings at or above it yield ``valid=True``;
                with ``higher_is_better=False``, ratings at or below it pass.
            higher_is_better: Whether higher ratings mean better text. Defaults to
                ``True``.
            model_id: Optional HuggingFace model ID; must be one of
                ``SUPPORTED_MODELS``. Defaults to
                ``opencompass/CompassJudger-2-7B-Instruct``.
            provider: Optional pre-configured provider (e.g. a ``LlamafileProvider``
                or a customized ``HuggingFaceProvider``). Defaults to a
                ``HuggingFaceProvider`` loading a causal LM. HuggingFace-backed loads
                force the SDPA attention kernel because CompassJudger-2's config
                requests flash_attention_2, which is unavailable on CPU/MPS.
            prompt: Optional prompt-template override, used as-is (must fill ``{criteria}`` /
                ``{rubric}`` / ``{instruction}`` / ``{response}``). Defaults to ``None`` — the
                registry default, or the version named by ``prompt_version``.
            prompt_version: Registered prompt version to use when ``prompt`` is not given. Defaults
                to ``None`` (the default version). See ``AnyGuardrail.list_prompt_versions``.

        Raises:
            ValueError: If ``model_id`` is not in ``SUPPORTED_MODELS``.

        """
        self.model_id = default(model_id, self.SUPPORTED_MODELS)
        self.criteria = criteria
        self.rubric = rubric
        self.pass_threshold = pass_threshold
        self.higher_is_better = higher_is_better
        self._prompt = resolve_prompt(GuardrailName.COMPASS_JUDGER, prompt, prompt_version)
        load_kwargs: AnyDict = {}
        # CompassJudger-2's config requests flash_attention_2, which isn't available on
        # CPU/MPS (or any env without flash-attn installed). Force the portable SDPA kernel.
        if provider is not None:
            self.provider = provider
            if isinstance(self.provider, HuggingFaceProvider):
                from transformers import AutoModelForCausalLM, AutoTokenizer

                load_kwargs = {
                    "model_class": AutoModelForCausalLM,
                    "tokenizer_class": AutoTokenizer,
                    "attn_implementation": "sdpa",
                }
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.provider = HuggingFaceProvider(model_class=AutoModelForCausalLM, tokenizer_class=AutoTokenizer)
            load_kwargs = {"attn_implementation": "sdpa"}
        self.provider.load_model(self.model_id, **load_kwargs)

    def validate(  # type: ignore[override]
        self, input_text: str | list[str], output_text: str | list[str] | None = None, **kwargs: Any
    ) -> GuardrailOutput | list[GuardrailOutput]:
        """Judge ``output_text`` (the response) given ``input_text`` (the instruction).

        Args:
            input_text: The instruction/prompt the response answers, e.g.
                ``"Summarize the article in two sentences."``, or a ``list[str]`` to
                judge a batch in one call.
            output_text: The response being judged — semantically the main text under
                evaluation. When ``None``, ``input_text`` itself is placed in the
                response slot of the judging prompt and judged directly. For a batched
                ``input_text``, this may be a matching-length list, a single value
                broadcast to every item, or omitted.
            **kwargs: Forwarded to the base pipeline; ignored by pre-processing.

        Returns:
            GuardrailOutput (or a list, one per item, in order, for a batched call)
            where ``valid`` maps the 1-10 rating through ``pass_threshold``, ``score``
            is the rating normalized onto the canonical risk axis (higher = riskier),
            ``explanation`` is the judge's justification, and ``extra["rubric_score"]``
            is the raw rating. Fails closed (``valid=False`` with
            ``extra={"parse_failure": True}``) when no rating parses.

        """
        return super().validate(input_text, output_text=output_text, **kwargs)

    def _build_messages(self, input_text: str, output_text: str | None) -> ChatMessages:
        prompt = self._prompt.segments["user"].format(
            criteria=self.criteria,
            rubric=self.rubric,
            instruction=input_text,
            response=output_text if output_text is not None else input_text,
        )
        return [{"role": "user", "content": prompt}]

    def _pre_processing(
        self, input_text: str, output_text: str | None = None, **kwargs: Any
    ) -> GuardrailPreprocessOutput[CompassJudgerPreprocessData]:
        """Fill the pointwise judging prompt and shape it as a single-turn chat.

        Args:
            input_text: The instruction/prompt the response answers.
            output_text: The response being judged; when ``None``, ``input_text`` is
                judged directly by placing it in the response slot.
            **kwargs: Ignored; accepted for signature compatibility.

        Returns:
            A ``GuardrailPreprocessOutput`` whose data holds the ``messages`` list for
            ``provider.generate_chat``.

        """
        del kwargs
        return GuardrailPreprocessOutput(data={"messages": self._build_messages(input_text, output_text)})

    def _inference(
        self, model_inputs: GuardrailPreprocessOutput[CompassJudgerPreprocessData]
    ) -> GuardrailInferenceOutput[CompassJudgerInferenceData]:
        return self.provider.generate_chat(
            messages=model_inputs.data["messages"], max_new_tokens=MAX_NEW_TOKENS, do_sample=False
        )

    def _build_result(self, text: str, prompt_tokens: int | None, completion_tokens: int | None) -> GuardrailOutput:
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
            usage=GuardrailUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
        )

    def _post_processing(self, model_outputs: GuardrailInferenceOutput[CompassJudgerInferenceData]) -> GuardrailOutput:
        """Parse the last ``Rating: [[X]]`` from the generation into a GuardrailOutput.

        ``valid`` maps the rating through ``pass_threshold``; ``score`` is the rating
        normalized onto the canonical risk axis (higher = riskier);
        ``extra["rubric_score"]`` is the raw integer. Fails closed
        (``valid=False`` with ``extra={"parse_failure": True}``) when no in-range
        rating is found.
        """
        data = model_outputs.data
        return self._build_result(
            data["generated_text"], data.get("prompt_token_count"), data.get("completion_token_count")
        )

    def _validate_batch(
        self, input_texts: list[str], output_text: str | list[str] | None = None, **kwargs: Any
    ) -> list[GuardrailOutput]:
        if not input_texts:
            return []
        if not isinstance(self.provider, HuggingFaceProvider):
            return super()._validate_batch(input_texts, output_text=output_text, **kwargs)
        outputs = broadcast_optional(output_text, len(input_texts), "output_text")
        batch = [self._build_messages(t, o) for t, o in zip(input_texts, outputs, strict=True)]
        result = self.provider.generate_chat(messages=batch, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        data = result.data
        return [
            self._build_result(text, prompt_tokens, completion_tokens)
            for text, prompt_tokens, completion_tokens in zip(
                data["generated_text"], data["prompt_token_count"], data["completion_token_count"], strict=True
            )
        ]
