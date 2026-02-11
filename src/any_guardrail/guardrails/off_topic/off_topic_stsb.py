import warnings
from typing import Any, ClassVar

import torch
from transformers import AutoModel, AutoTokenizer

from any_guardrail.base import GuardrailOutput, ThreeStageGuardrail
from any_guardrail.guardrails.off_topic.models.cross_encoder_mlp import CrossEncoderWithMLP
from any_guardrail.guardrails.utils import default
from any_guardrail.providers.base import StandardProvider
from any_guardrail.types import GuardrailInferenceOutput, GuardrailPreprocessOutput

BASEMODEL = "cross-encoder/stsb-roberta-base"

# Type aliases for OffTopicStsb
StsbPreprocessData = tuple[torch.Tensor, torch.Tensor]
StsbInferenceData = Any  # Model output tensor


class OffTopicStsb(ThreeStageGuardrail[StsbPreprocessData, StsbInferenceData, bool, dict[str, float], float]):
    """Wrapper for off-topic detection model from govtech.

    For more information, please see the model card:

    - [govtech/stsb-roberta-base-off-topic model](https://huggingface.co/govtech/stsb-roberta-base-off-topic).
    """

    SUPPORTED_MODELS: ClassVar = ["mozilla-ai/stsb-roberta-base-off-topic"]

    def __init__(
        self,
        model_id: str | None = None,
        provider: StandardProvider | None = None,  # Reserved for future extensibility
    ) -> None:
        """Initialize the OffTopicStsb guardrail."""
        self.model_id = default(model_id, self.SUPPORTED_MODELS)
        self.provider = provider  # Reserved for future extensibility

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(BASEMODEL)
        base_model = AutoModel.from_pretrained(BASEMODEL)
        self.model = CrossEncoderWithMLP.from_pretrained(self.model_id, base_model=base_model)

    def _pre_processing(
        self, input_text: str, comparison_text: str | None = None
    ) -> GuardrailPreprocessOutput[StsbPreprocessData]:
        warnings.warn("Truncating text to a maximum length of 514 tokens.", stacklevel=2)
        encoding = self.tokenizer(
            input_text,
            comparison_text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=514,
            return_token_type_ids=False,
        )
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        return GuardrailPreprocessOutput(data=(input_ids, attention_mask))

    def _inference(
        self,
        model_inputs: GuardrailPreprocessOutput[StsbPreprocessData],
    ) -> GuardrailInferenceOutput[StsbInferenceData]:
        """Run cross-encoder inference with separate input_ids and attention_mask."""
        data = model_inputs.data
        input_ids, attention_mask = data
        with torch.no_grad():
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return GuardrailInferenceOutput(data=output)

    def _post_processing(
        self, model_outputs: GuardrailInferenceOutput[StsbInferenceData]
    ) -> GuardrailOutput[bool, dict[str, float], float]:
        probabilities = torch.softmax(model_outputs.data, dim=1)
        predicted_label = torch.argmax(probabilities, dim=1).item()
        explanatory_probs = probabilities.cpu().numpy().tolist()[0]
        probs_dict = {"on-topic": explanatory_probs[0], "off-topic": explanatory_probs[1]}

        return GuardrailOutput(
            valid=predicted_label != 1,  # Assuming label '1' indicates off-topic
            explanation=probs_dict,
        )
