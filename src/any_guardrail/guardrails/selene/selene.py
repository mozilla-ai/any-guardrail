import re
from typing import Any, ClassVar

from any_guardrail.base import GuardrailName, GuardrailOutput, ThreeStageGuardrail
from any_guardrail.guardrails.utils import default, normalize_rubric_to_risk
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

SelenePreprocessData = AnyDict
SeleneInferenceData = AnyDict

SELENE_PROMPT = """You are tasked with evaluating a response based on a given instruction (which may contain an Input) and a scoring rubric that serve as the evaluation standard. Provide a comprehensive feedback on the response quality strictly adhering to the scoring rubric, without any general evaluation. Follow this with a score between 1 and 5, referring to the scoring rubric. Avoid generating any additional opening, closing, or explanations.

Here are some rules of the evaluation:
(1) You should prioritize evaluating whether the response satisfies the provided rubric. The basis of your score should depend exactly on the rubric. However, the response does not need to explicitly address points raised in the rubric. Rather, evaluate the response based on the criteria outlined in the rubric.

Your reply should strictly follow this format:
**Reasoning:** <Your feedback>

**Result:** <an integer between 1 and 5>

Here is the data:

Instruction:
```
{instruction}
```

Response:
```
{response}
```

Score Rubrics:
{rubric}
"""

SCORE_MIN = 1
SCORE_MAX = 5
MAX_NEW_TOKENS = 1024

_RESULT_PATTERN = re.compile(r"\*\*Result:\*\*\s*([1-5])\b")


class Selene(ThreeStageGuardrail[SelenePreprocessData, SeleneInferenceData]):
    """Selene 1 Mini — general-purpose LLM judge grading a response against a user-defined 1-5 rubric (Atla).

    Selene 1 Mini is an 8B evaluator LLM (fine-tuned from Llama 3.1 8B) specialized in
    scoring model outputs. This guardrail drives it in single-rubric absolute-grading mode:
    each call wraps the instruction and response in Selene's evaluation prompt together with
    the caller's ``rubric``, and the model replies with a ``**Reasoning:**`` block followed
    by ``**Result:** <n>`` where ``n`` is an integer 1-5. It runs through
    ``provider.generate_chat``, so it can be served from either a ``HuggingFaceProvider`` or
    a ``LlamafileProvider``.

    Verdict mapping onto ``GuardrailOutput``:

    - ``valid`` is ``rubric_score >= pass_threshold`` (or ``<=`` when
      ``higher_is_better=False``).
    - ``score`` (canonical risk: higher = riskier) is the 1-5 rubric score normalized onto
      [0, 1] via ``normalize_rubric_to_risk`` — inverted when higher rubric values mean
      better, so a high-quality response yields a low risk.
    - ``explanation`` is the model's full generation (reasoning plus the result line).
    - ``extra["rubric_score"]`` is the raw integer 1-5.
    - When no ``**Result:**`` score can be parsed, the output fails closed: ``valid=False``
      with ``extra={"parse_failure": True}``.

    Inputs are single strings: ``input_text`` is the instruction and ``output_text`` is the
    response being graded (when ``output_text`` is omitted, ``input_text`` is graded as the
    response). List/batch input is not supported.

    For more information, see:

    - [Selene-1-Mini-Llama-3.1-8B model card](https://huggingface.co/AtlaAI/Selene-1-Mini-Llama-3.1-8B)
    - [Atla Selene Mini: A General Purpose Evaluation Model (arXiv:2501.17195)](https://arxiv.org/abs/2501.17195)
    - [Selene 1 Mini announcement (Atla)](https://www.atla-ai.com/post/selene-1-mini)

    Args:
        rubric: The score rubric — the evaluation objective plus ``Score 1:`` … ``Score 5:``
            descriptions of what each level means.
        pass_threshold: The score (1-5) at or above which the response passes (or at or below
            when ``higher_is_better=False``).
        higher_is_better: Whether higher rubric scores mean better responses. Defaults to
            ``True``.
        model_id: Optional HuggingFace model ID; must be one of ``SUPPORTED_MODELS``.
            Defaults to ``AtlaAI/Selene-1-Mini-Llama-3.1-8B``.
        provider: Optional pre-configured provider. Defaults to a ``HuggingFaceProvider``
            loading a causal LM.

    """

    SUPPORTED_MODELS: ClassVar = ["AtlaAI/Selene-1-Mini-Llama-3.1-8B"]

    METADATA: ClassVar[GuardrailMetadata] = GUARDRAIL_METADATA[GuardrailName.SELENE]

    def __init__(
        self,
        rubric: str,
        pass_threshold: int,
        higher_is_better: bool = True,
        model_id: str | None = None,
        provider: StandardProvider | None = None,
    ) -> None:
        """Initialize the Selene guardrail.

        Args:
            rubric: The score rubric applied to every ``validate`` call — the evaluation
                objective plus ``Score 1:`` … ``Score 5:`` descriptions, e.g.
                ``"How helpful is the answer? Score 1: not helpful. ... Score 5: fully helpful."``.
            pass_threshold: The score (1-5) at or above which the response passes (or at or
                below when ``higher_is_better=False``), e.g. ``4``.
            higher_is_better: Whether higher rubric scores mean better responses. Set
                ``False`` for rubrics where a higher number is worse. Defaults to ``True``.
            model_id: Optional HuggingFace model ID; must be one of ``SUPPORTED_MODELS``.
                Defaults to ``AtlaAI/Selene-1-Mini-Llama-3.1-8B``.
            provider: Optional pre-configured provider. When ``None``, a
                ``HuggingFaceProvider`` is built targeting ``AutoModelForCausalLM`` /
                ``AutoTokenizer`` (transformers is imported lazily here). Pass a
                ``LlamafileProvider`` to run a GGUF build without the huggingface extra.

        Raises:
            ValueError: If ``model_id`` is not in ``SUPPORTED_MODELS``.

        """
        self.model_id = default(model_id, self.SUPPORTED_MODELS)
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
        """Judge ``output_text`` (the response) given ``input_text`` (the instruction).

        Args:
            input_text: The instruction the response was produced for, e.g.
                ``"Summarize the article in one sentence."``. Single string only; list/batch
                input is not supported and raises ``TypeError``.
            output_text: The response being graded against the rubric, e.g.
                ``"The article argues that remote work boosts productivity."``. When ``None``,
                ``input_text`` itself is graded as the response.
            **kwargs: Ignored; accepted only so the signature matches the ``ThreeStageGuardrail``
                contract.

        Returns:
            GuardrailOutput where ``valid`` maps the rubric score through ``pass_threshold``,
            ``score`` is the 1-5 rubric score normalized onto the canonical risk axis
            (higher = riskier), ``explanation`` is the model's reasoning, and
            ``extra["rubric_score"]`` is the raw integer. When no score can be parsed the
            output fails closed (``valid=False`` with ``extra={"parse_failure": True}``).

        """
        result = super().validate(input_text, output_text=output_text, **kwargs)
        if isinstance(result, list):
            msg = "Selene.validate received a list input but only supports single strings."
            raise TypeError(msg)
        return result

    def _pre_processing(
        self, input_text: str, output_text: str | None = None, **kwargs: Any
    ) -> GuardrailPreprocessOutput[SelenePreprocessData]:
        """Format Selene's single-rubric evaluation prompt as a one-turn user chat message.

        Args:
            input_text: The instruction, substituted into the prompt's ``Instruction`` block.
            output_text: The response to grade, substituted into the ``Response`` block. When
                ``None``, ``input_text`` is reused as the response.
            **kwargs: Ignored (dropped via ``del kwargs``); present only for signature
                compatibility with the ``ThreeStageGuardrail`` contract.

        Returns:
            GuardrailPreprocessOutput wrapping the one-message chat prompt under the
            ``messages`` key.

        """
        del kwargs
        prompt = SELENE_PROMPT.format(
            instruction=input_text,
            response=output_text if output_text is not None else input_text,
            rubric=self.rubric,
        )
        messages: ChatMessages = [{"role": "user", "content": prompt}]
        return GuardrailPreprocessOutput(data={"messages": messages})

    def _inference(
        self, model_inputs: GuardrailPreprocessOutput[SelenePreprocessData]
    ) -> GuardrailInferenceOutput[SeleneInferenceData]:
        return self.provider.generate_chat(
            messages=model_inputs.data["messages"], max_new_tokens=MAX_NEW_TOKENS, do_sample=False
        )

    def _post_processing(self, model_outputs: GuardrailInferenceOutput[SeleneInferenceData]) -> GuardrailOutput:
        text = model_outputs.data["generated_text"]
        match = _RESULT_PATTERN.search(text)
        if match is None:
            return GuardrailOutput(valid=False, explanation=text, extra={"parse_failure": True})
        rubric_score = int(match.group(1))
        passed = rubric_score >= self.pass_threshold if self.higher_is_better else rubric_score <= self.pass_threshold
        return GuardrailOutput(
            valid=passed,
            score=normalize_rubric_to_risk(rubric_score, SCORE_MIN, SCORE_MAX, higher_is_better=self.higher_is_better),
            explanation=text,
            extra={"rubric_score": rubric_score},
            usage=GuardrailUsage(
                prompt_tokens=model_outputs.data.get("prompt_token_count"),
                completion_tokens=model_outputs.data.get("completion_token_count"),
            ),
        )
