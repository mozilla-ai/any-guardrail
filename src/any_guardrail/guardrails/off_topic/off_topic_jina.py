import warnings
from typing import Any, ClassVar

import torch
from transformers import AutoModel, AutoTokenizer

from any_guardrail.base import GuardrailOutput, ThreeStageGuardrail
from any_guardrail.guardrails.off_topic.models.cross_encoder_shared import CrossEncoderWithSharedBase
from any_guardrail.providers.base import Provider
from any_guardrail.types import GuardrailInferenceOutput, GuardrailPreprocessOutput

BASEMODEL = "jinaai/jina-embeddings-v2-small-en"

# Type aliases for OffTopicJina
JinaPreprocessData = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
JinaInferenceData = Any  # Model output tensor


class OffTopicJinaProvider(Provider[JinaPreprocessData, JinaInferenceData]):
    """Provider for OffTopicJina cross-encoder model."""

    def __init__(self) -> None:
        """Initialize the OffTopicJina provider."""
        self.model: Any = None
        self.tokenizer: Any = None

    def load_model(self, model_id: str, **kwargs: Any) -> None:
        """Load base model and custom cross-encoder with shared base."""
        self.tokenizer = AutoTokenizer.from_pretrained(BASEMODEL)
        base_model = AutoModel.from_pretrained(BASEMODEL)
        self.model = CrossEncoderWithSharedBase.from_pretrained(model_id, base_model=base_model)

    def pre_process(self, *args: Any, **kwargs: Any) -> GuardrailPreprocessOutput[JinaPreprocessData]:
        """Preprocessing is handled by the guardrail."""
        msg = "OffTopicJina preprocessing is handled by the guardrail."
        raise NotImplementedError(msg)

    def infer(
        self, model_inputs: GuardrailPreprocessOutput[JinaPreprocessData]
    ) -> GuardrailInferenceOutput[JinaInferenceData]:
        """Run cross-encoder inference with four separate tensor inputs."""
        data = model_inputs.data
        if len(data) != 4:
            msg = "Expected model_inputs to be a tuple of (input_ids1, attention_mask1, input_ids2, attention_mask2)."
            raise ValueError(msg)
        input_ids1, attention_mask1, input_ids2, attention_mask2 = data
        with torch.no_grad():
            output = self.model(
                input_ids1=input_ids1,
                attention_mask1=attention_mask1,
                input_ids2=input_ids2,
                attention_mask2=attention_mask2,
            )
        return GuardrailInferenceOutput(data=output)


class OffTopicJina(ThreeStageGuardrail[JinaPreprocessData, JinaInferenceData, bool, dict[str, float], float]):
    """Wrapper for off-topic detection model from govtech.

    For more information, please see the model card:

    - [govtech/jina-embeddings-v2-small-en-off-topic](https://huggingface.co/govtech/jina-embeddings-v2-small-en-off-topic).
    """

    SUPPORTED_MODELS: ClassVar = ["mozilla-ai/jina-embeddings-v2-small-en-off-topic"]

    def __init__(
        self,
        model_id: str | None = None,
        provider: OffTopicJinaProvider | None = None,
    ) -> None:
        """Initialize the OffTopicJina guardrail."""
        self.model_id = model_id or self.SUPPORTED_MODELS[0]
        if self.model_id not in self.SUPPORTED_MODELS:
            msg = f"Only supports {self.SUPPORTED_MODELS}. Please use this path to instantiate model."
            raise ValueError(msg)
        self.provider = provider or OffTopicJinaProvider()
        self.provider.load_model(self.model_id)

    def validate(
        self, input_text: str, comparison_text: str | None = None
    ) -> GuardrailOutput[bool, dict[str, float], float]:
        """Validate whether the input text is on or off topic."""
        model_inputs = self._pre_processing(input_text, comparison_text)
        model_outputs = self._inference(model_inputs)
        return self._post_processing(model_outputs)

    def _pre_processing(
        self, input_text: str, comparison_text: str | None = None
    ) -> GuardrailPreprocessOutput[JinaPreprocessData]:
        warnings.warn("Truncating input text to a max length of 1024 tokens.", stacklevel=2)
        inputs1 = self.provider.tokenizer(
            input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=1024
        )
        inputs2 = self.provider.tokenizer(
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
        return self.provider.infer(model_inputs)

    def _post_processing(
        self, model_outputs: GuardrailInferenceOutput[JinaInferenceData]
    ) -> GuardrailOutput[bool, dict[str, float], float]:
        probabilities = torch.softmax(model_outputs.data, dim=1)
        predicted_label = torch.argmax(probabilities, dim=1).item()
        explanatory_probs = probabilities.cpu().numpy().tolist()[0]
        probs_dict = {"on-topic": explanatory_probs[0], "off-topic": explanatory_probs[1]}

        return GuardrailOutput(
            valid=predicted_label != 1,  # Assuming label '1' indicates off-topic
            explanation=probs_dict,
        )
