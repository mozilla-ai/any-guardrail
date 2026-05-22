import os
from typing import Any, ClassVar

try:
    from openai import OpenAI
except ImportError as e:
    msg = (
        "openai package is not installed. "
        "Please install it with `pip install openai` to use the OpenaiModeration guardrail."
    )
    raise ImportError(msg) from e

from any_guardrail.base import GuardrailOutput, ThreeStageGuardrail
from any_guardrail.guardrails.utils import default
from any_guardrail.types import AnyDict, GuardrailInferenceOutput, GuardrailPreprocessOutput


class OpenaiModeration(ThreeStageGuardrail[AnyDict, Any, bool, dict[str, float], float]):
    """Guardrail implementation using OpenAI's Moderation API.

    Wraps OpenAI's hosted moderation classifier (default model:
    ``omni-moderation-latest``) to flag content across 13 harm
    sub-categories: hate, hate/threatening, harassment, harassment/threatening,
    self-harm, self-harm/intent, self-harm/instructions, sexual, sexual/minors,
    violence, violence/graphic, illicit, and illicit/violent.

    The classifier returns a calibrated per-category probability score
    alongside a boolean ``flagged`` verdict. This guardrail surfaces both:
    ``valid`` is ``False`` when OpenAI flags the content **or** when the
    maximum per-category score exceeds ``threshold``; ``explanation`` is the
    full ``{category: score}`` dict; ``score`` is the max category score.

    The current ``omni-moderation`` model is a GPT-4o-derived multimodal
    classifier; the original methodology (taxonomy, active-learning loop,
    calibration) is described in Markov et al. 2023,
    [A Holistic Approach to Undesired Content Detection in the Real World](https://arxiv.org/abs/2208.03274)
    (AAAI 2023). See the
    [omni-moderation announcement](https://openai.com/index/upgrading-the-moderation-api-with-our-new-multimodal-moderation-model/)
    for the multimodal upgrade and the
    [moderation guide](https://platform.openai.com/docs/guides/moderation)
    for usage details. The Moderation API is free and does not count toward
    standard usage quotas.

    """

    SUPPORTED_MODELS: ClassVar = [
        "omni-moderation-latest",
        "omni-moderation-2024-09-26",
        "text-moderation-latest",
    ]

    def __init__(
        self,
        model_id: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        threshold: float = 0.5,
    ) -> None:
        """Initialize the OpenAI Moderation guardrail.

        Args:
            model_id (str | None): The moderation model to use. Defaults to
                ``omni-moderation-latest``.
            api_key (str | None): OpenAI API key. Falls back to the
                ``OPENAI_API_KEY`` environment variable when not provided.
            base_url (str | None): Optional custom base URL for the OpenAI
                client (useful for proxies or Azure-style routing).
            threshold (float): Maximum per-category score above which content
                is considered invalid even if OpenAI did not explicitly flag
                it. Defaults to ``0.5``.

        """
        self.model_id = default(model_id, self.SUPPORTED_MODELS)

        resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_api_key:
            msg = (
                "OpenAI API key not provided. Pass `api_key=...` to the "
                "constructor or set the OPENAI_API_KEY environment variable."
            )
            raise ValueError(msg)

        self.threshold = threshold
        client_kwargs: dict[str, Any] = {"api_key": resolved_api_key}
        if base_url is not None:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)

    def _pre_processing(self, text: str) -> GuardrailPreprocessOutput[AnyDict]:
        """Wrap the input string into the OpenAI moderation request payload."""
        return GuardrailPreprocessOutput(data={"input": text})

    def _inference(self, model_inputs: GuardrailPreprocessOutput[AnyDict]) -> GuardrailInferenceOutput[Any]:
        """Call the OpenAI Moderation API and return the raw SDK response."""
        response = self.client.moderations.create(model=self.model_id, input=model_inputs.data["input"])
        return GuardrailInferenceOutput(data=response)

    def _post_processing(
        self, model_outputs: GuardrailInferenceOutput[Any]
    ) -> GuardrailOutput[bool, dict[str, float], float]:
        """Translate the OpenAI moderation result into a GuardrailOutput.

        Returns:
            GuardrailOutput where ``valid`` is False if OpenAI flagged the
            content or the max category score exceeds ``self.threshold``,
            ``explanation`` is the full ``{category: score}`` dict, and
            ``score`` is the max category score.

        """
        result = model_outputs.data.results[0]
        category_scores = result.category_scores
        if hasattr(category_scores, "model_dump"):
            scores_dict: dict[str, float] = category_scores.model_dump()
        elif isinstance(category_scores, dict):
            scores_dict = dict(category_scores)
        else:
            scores_dict = dict(vars(category_scores))

        filtered_scores: dict[str, float] = {k: float(v) for k, v in scores_dict.items() if v is not None}
        max_score: float = max(filtered_scores.values()) if filtered_scores else 0.0
        flagged = bool(result.flagged)
        valid = not flagged and max_score < self.threshold
        return GuardrailOutput(valid=valid, explanation=filtered_scores, score=max_score)
