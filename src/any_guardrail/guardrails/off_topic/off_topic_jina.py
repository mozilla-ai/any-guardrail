import warnings
from typing import Any, ClassVar

import torch
from transformers import AutoModel, AutoTokenizer

from any_guardrail.base import GuardrailOutput, ThreeStageGuardrail
from any_guardrail.guardrails.off_topic._postprocess import off_topic_output
from any_guardrail.guardrails.off_topic.models.cross_encoder_shared import CrossEncoderWithSharedBase
from any_guardrail.guardrails.utils import default
from any_guardrail.providers.base import StandardProvider
from any_guardrail.types import GuardrailInferenceOutput, GuardrailPreprocessOutput

BASEMODEL = "jinaai/jina-embeddings-v2-small-en"

# Type aliases for OffTopicJina
JinaPreprocessData = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
JinaInferenceData = Any  # Model output tensor


class OffTopicJina(ThreeStageGuardrail[JinaPreprocessData, JinaInferenceData]):
    """Off-Topic (Jina bi-encoder) — GovTech off-topic relevance detector using dual jina-embeddings-v2-small-en encoders.

    The bi-encoder implementation dispatched by ``OffTopic`` for
    ``mozilla-ai/jina-embeddings-v2-small-en-off-topic``. It embeds ``input_text`` and
    ``comparison_text`` separately with a shared jina-embeddings-v2-small-en base, then
    learns their relationship through cross-attention layers before a 2-class head. Both
    inputs are truncated to 1024 tokens (a ``warnings.warn`` fires when this happens).
    Output maps to ``GuardrailOutput`` exactly as ``OffTopic`` documents: ``valid`` is
    ``True`` on-topic, ``score`` is ``P(off-topic)`` (higher = riskier), and
    ``categories`` reports both class probabilities. English-language model.

    For more information, see:

    - [Off-Topic (Jina bi-encoder) model card](https://huggingface.co/mozilla-ai/jina-embeddings-v2-small-en-off-topic).
    - [govtech/jina-embeddings-v2-small-en-off-topic](https://huggingface.co/govtech/jina-embeddings-v2-small-en-off-topic) (upstream).

    Args:
        model_id: Optional HuggingFace model ID. Must be one of ``SUPPORTED_MODELS``;
            defaults to ``mozilla-ai/jina-embeddings-v2-small-en-off-topic``.
        provider: Reserved for future extensibility; currently unused. The model is loaded
            directly via ``transformers``.

    """

    SUPPORTED_MODELS: ClassVar = ["mozilla-ai/jina-embeddings-v2-small-en-off-topic"]

    def __init__(
        self,
        model_id: str | None = None,
        provider: StandardProvider | None = None,  # Reserved for future extensibility
    ) -> None:
        """Initialize the OffTopicJina guardrail.

        Args:
            model_id: Optional HuggingFace model ID. Must be one of ``SUPPORTED_MODELS``;
                defaults to ``mozilla-ai/jina-embeddings-v2-small-en-off-topic``.
            provider: Reserved for future extensibility; currently unused. The
                cross-encoder is loaded directly via ``transformers``.

        Raises:
            ValueError: If ``model_id`` is not in ``SUPPORTED_MODELS``.

        """
        self.model_id = default(model_id, self.SUPPORTED_MODELS)
        self.provider = provider  # Reserved for future extensibility

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(BASEMODEL)
        base_model = AutoModel.from_pretrained(BASEMODEL)
        self.model = CrossEncoderWithSharedBase.from_pretrained(self.model_id, base_model=base_model)

    def _pre_processing(
        self, input_text: str, comparison_text: str | None = None
    ) -> GuardrailPreprocessOutput[JinaPreprocessData]:
        """Tokenize both texts separately for the bi-encoder.

        Args:
            input_text: The text being classified; tokenized as the first sequence
                (truncated / padded to 1024 tokens).
            comparison_text: The reference topic; tokenized as the second sequence
                (truncated / padded to 1024 tokens).

        Returns:
            GuardrailPreprocessOutput wrapping the four tensors
            (``input_ids`` / ``attention_mask`` for each text) the bi-encoder consumes.

        """
        warnings.warn("Truncating input text to a max length of 1024 tokens.", stacklevel=2)
        inputs1 = self.tokenizer(
            input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=1024
        )
        inputs2 = self.tokenizer(
            comparison_text, return_tensors="pt", truncation=True, padding="max_length", max_length=1024
        )
        input_ids1 = inputs1["input_ids"]
        attention_mask1 = inputs1["attention_mask"]
        input_ids2 = inputs2["input_ids"]
        attention_mask2 = inputs2["attention_mask"]
        return GuardrailPreprocessOutput(data=(input_ids1, attention_mask1, input_ids2, attention_mask2))

    def _inference(
        self,
        model_inputs: GuardrailPreprocessOutput[JinaPreprocessData],
    ) -> GuardrailInferenceOutput[JinaInferenceData]:
        """Run cross-encoder inference with four separate tensor inputs."""
        data = model_inputs.data
        input_ids1, attention_mask1, input_ids2, attention_mask2 = data
        with torch.no_grad():
            output = self.model(
                input_ids1=input_ids1,
                attention_mask1=attention_mask1,
                input_ids2=input_ids2,
                attention_mask2=attention_mask2,
            )
        return GuardrailInferenceOutput(data=output)

    def _post_processing(self, model_outputs: GuardrailInferenceOutput[JinaInferenceData]) -> GuardrailOutput:
        return off_topic_output(model_outputs.data)
