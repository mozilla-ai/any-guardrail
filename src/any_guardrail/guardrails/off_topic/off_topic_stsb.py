import warnings
from typing import Any, ClassVar

import torch
from transformers import AutoModel, AutoTokenizer

from any_guardrail.base import GuardrailOutput
from any_guardrail.guardrails.huggingface import HuggingFace
from any_guardrail.guardrails.off_topic.models.cross_encoder_mlp import CrossEncoderWithMLP
from any_guardrail.types import GuardrailInferenceOutput, GuardrailPreprocessOutput

BASEMODEL = "cross-encoder/stsb-roberta-base"

# Type aliases for OffTopicStsb
StsbPreprocessData = tuple[torch.Tensor, torch.Tensor]
StsbInferenceData = Any  # Model output tensor


class OffTopicStsb(HuggingFace[StsbPreprocessData, StsbInferenceData, bool, dict[str, float], float]):
    """Wrapper for off-topic detection model from govtech.

    For more information, please see the model card:

    - [govtech/stsb-roberta-base-off-topic model](https://huggingface.co/govtech/stsb-roberta-base-off-topic).
    """

    SUPPORTED_MODELS: ClassVar = ["mozilla-ai/stsb-roberta-base-off-topic"]

    def _load_model(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(BASEMODEL)  # type: ignore[no-untyped-call]
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
        input_ids = encoding["input_ids"]  # .to(device)
        attention_mask = encoding["attention_mask"]  # .to(device)
        return GuardrailPreprocessOutput(data=(input_ids, attention_mask))

    def _inference(
        self,
        model_inputs: GuardrailPreprocessOutput[StsbPreprocessData],
    ) -> GuardrailInferenceOutput[StsbInferenceData]:
        data = model_inputs.data
        if len(data) != 2:
            msg = "Expected model_inputs to be a tuple of (input_ids, attention_mask)."
            raise ValueError(msg)
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
