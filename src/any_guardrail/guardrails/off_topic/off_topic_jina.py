import warnings
from typing import Any, ClassVar

import torch
from transformers import AutoModel, AutoTokenizer

from any_guardrail.base import GuardrailOutput, ThreeStageGuardrail
from any_guardrail.guardrails.off_topic.models.cross_encoder_shared import CrossEncoderWithSharedBase
from any_guardrail.guardrails.utils import default
from any_guardrail.providers.base import StandardProvider
from any_guardrail.types import GuardrailInferenceOutput, GuardrailPreprocessOutput

BASEMODEL = "jinaai/jina-embeddings-v2-small-en"

# Type aliases for OffTopicJina
JinaPreprocessData = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
JinaInferenceData = Any  # Model output tensor


class OffTopicJina(ThreeStageGuardrail[JinaPreprocessData, JinaInferenceData, bool, dict[str, float], float]):
    """Wrapper for off-topic detection model from govtech.

    For more information, please see the model card:

    - [govtech/jina-embeddings-v2-small-en-off-topic](https://huggingface.co/govtech/jina-embeddings-v2-small-en-off-topic).
    """

    SUPPORTED_MODELS: ClassVar = ["mozilla-ai/jina-embeddings-v2-small-en-off-topic"]

    def __init__(
        self,
        model_id: str | None = None,
        provider: StandardProvider | None = None,  # Reserved for future extensibility
    ) -> None:
        """Initialize the OffTopicJina guardrail."""
        self.model_id = default(model_id, self.SUPPORTED_MODELS)
        self.provider = provider  # Reserved for future extensibility

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(BASEMODEL)
        base_model = AutoModel.from_pretrained(BASEMODEL)
        self.model = CrossEncoderWithSharedBase.from_pretrained(self.model_id, base_model=base_model)

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
