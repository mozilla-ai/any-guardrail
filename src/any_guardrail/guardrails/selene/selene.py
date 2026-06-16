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
    """Atla Selene-1-Mini — general-purpose LLM judge.

    Evaluates a response against a user-defined ``rubric`` on a 1-5 scale and returns
    reasoning plus a score. ``valid`` maps the score through ``pass_threshold``;
    ``score`` is the rubric normalized onto the canonical risk axis; ``explanation`` is
    the model's reasoning. Fails closed (``valid=False`` with
    ``extra={"parse_failure": True}``) when no score parses.

    For more information, see the
    [Selene-1-Mini model card](https://huggingface.co/AtlaAI/Selene-1-Mini-Llama-3.1-8B).

    Args:
        rubric: The score rubric (objective plus ``Score 1:`` … ``Score 5:`` descriptions).
        pass_threshold: The score (1-5) at or above which the response passes.
        higher_is_better: Whether higher scores mean better. Defaults to ``True``.
        model_id: Optional HuggingFace model ID. Defaults to ``AtlaAI/Selene-1-Mini-Llama-3.1-8B``.
        provider: Optional pre-configured provider. Defaults to a ``HuggingFaceProvider``
            loading a causal LM.
    """

    SUPPORTED_MODELS: ClassVar = ["AtlaAI/Selene-1-Mini-Llama-3.1-8B"]

    def __init__(
        self,
        rubric: str,
        pass_threshold: int,
        higher_is_better: bool = True,
        model_id: str | None = None,
        provider: StandardProvider | None = None,
    ) -> None:
        """Initialize the Selene guardrail."""
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
        """Judge ``output_text`` (the response) given ``input_text`` (the instruction)."""
        result = super().validate(input_text, output_text=output_text, **kwargs)
        if isinstance(result, list):
            msg = "Selene.validate received a list input but only supports single strings."
            raise TypeError(msg)
        return result

    def _pre_processing(
        self, input_text: str, output_text: str | None = None, **kwargs: Any
    ) -> GuardrailPreprocessOutput[SelenePreprocessData]:
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

    def _post_processing(
        self, model_outputs: GuardrailInferenceOutput[SeleneInferenceData]
    ) -> GuardrailOutput:
        text = model_outputs.data["generated_text"]
        match = _RESULT_PATTERN.search(text)
        if match is None:
            return GuardrailOutput(valid=False, explanation=text, extra={"parse_failure": True})
        rubric_score = int(match.group(1))
        passed = (
            rubric_score >= self.pass_threshold if self.higher_is_better else rubric_score <= self.pass_threshold
        )
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
