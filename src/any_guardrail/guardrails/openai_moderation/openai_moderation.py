import os
from typing import Any, ClassVar

from any_guardrail.base import GuardrailName
from any_guardrail.registry import GUARDRAIL_METADATA
from any_guardrail.taxonomy import GuardrailMetadata

try:
    from openai import OpenAI
except ImportError as e:
    msg = (
        "openai package is not installed. "
        "Please install it with `pip install 'any-guardrail[openai]'` to use the OpenaiModeration guardrail."
    )
    raise ImportError(msg) from e

from any_guardrail.base import GuardrailOutput, ThreeStageGuardrail
from any_guardrail.guardrails.utils import default
from any_guardrail.types import AnyDict, CategoryResult, GuardrailInferenceOutput, GuardrailPreprocessOutput


class OpenaiModeration(ThreeStageGuardrail[AnyDict, Any]):
    """OpenAI Moderation — hosted moderation API flagging content across 13 harm categories with calibrated scores (OpenAI).

    Wraps OpenAI's hosted moderation classifier (default model:
    ``omni-moderation-latest``) to flag content across 13 harm
    sub-categories: hate, hate/threatening, harassment, harassment/threatening,
    self-harm, self-harm/intent, self-harm/instructions, sexual, sexual/minors,
    violence, violence/graphic, illicit, and illicit/violent. The classifier
    returns a calibrated per-category probability score alongside a boolean
    ``flagged`` verdict.

    Expected input: a single string (or a list of strings, screened one at a time
    via the batched ``ThreeStageGuardrail.validate``).

    ``GuardrailOutput`` mapping:
        - ``valid`` is ``False`` when OpenAI flags the content **or** when the
          maximum per-category score exceeds ``threshold`` (otherwise ``True``).
        - ``score`` is the maximum per-category probability (higher = riskier).
        - ``categories`` is the full per-category breakdown: each
          ``CategoryResult`` carries the calibrated ``score`` and a ``triggered``
          flag (set when OpenAI flagged that category or its score exceeds
          ``threshold``).
        - ``raw`` is the full OpenAI SDK moderation response object.

    The current ``omni-moderation`` model is a GPT-4o-derived multimodal
    classifier; the original methodology (taxonomy, active-learning loop,
    calibration) is described in Markov et al. 2023 (AAAI 2023). The Moderation
    API is free and does not count toward standard usage quotas.

    For more information, see:

    - [Moderation guide (usage)](https://platform.openai.com/docs/guides/moderation)
    - [Upgrading the Moderation API with our new multimodal moderation model](https://openai.com/index/upgrading-the-moderation-api-with-our-new-multimodal-moderation-model/)
    - [A Holistic Approach to Undesired Content Detection in the Real World (arXiv:2208.03274)](https://arxiv.org/abs/2208.03274)

    """

    SUPPORTED_MODELS: ClassVar = [
        "omni-moderation-latest",
        "omni-moderation-2024-09-26",
        "text-moderation-latest",
    ]

    METADATA: ClassVar[GuardrailMetadata] = GUARDRAIL_METADATA[GuardrailName.OPENAI_MODERATION]

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

    def _post_processing(self, model_outputs: GuardrailInferenceOutput[Any]) -> GuardrailOutput:
        """Translate the OpenAI moderation result into a GuardrailOutput.

        Returns:
            GuardrailOutput where ``valid`` is False if OpenAI flagged the
            content or the max category score exceeds ``self.threshold``,
            ``categories`` is the full per-category breakdown (score +
            triggered flag), and ``score`` is the max category score.

        """
        result = model_outputs.data.results[0]
        scores_dict = self._unwrap(result.category_scores)
        flags_dict = self._unwrap(result.categories)

        filtered_scores: dict[str, float] = {k: float(v) for k, v in scores_dict.items() if v is not None}
        max_score: float = max(filtered_scores.values()) if filtered_scores else 0.0
        flagged = bool(result.flagged)
        valid = not flagged and max_score <= self.threshold

        categories = [
            CategoryResult(name=cat, score=score, triggered=bool(flags_dict.get(cat)) or score > self.threshold)
            for cat, score in filtered_scores.items()
        ]
        return GuardrailOutput(valid=valid, score=max_score, categories=categories, raw=model_outputs.data)

    @staticmethod
    def _unwrap(obj: Any) -> dict[str, Any]:
        """Coerce an OpenAI SDK per-category object into a plain dict."""
        if hasattr(obj, "model_dump"):
            return dict(obj.model_dump())
        if isinstance(obj, dict):
            return dict(obj)
        return dict(vars(obj))
