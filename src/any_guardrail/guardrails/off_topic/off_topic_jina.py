import warnings
from typing import Any, ClassVar

import torch
from transformers import AutoModel, AutoTokenizer

from any_guardrail.base import GuardrailOutput
from any_guardrail.guardrails.huggingface import HuggingFace
from any_guardrail.guardrails.off_topic.models.cross_encoder_shared import CrossEncoderWithSharedBase

BASEMODEL = "jinaai/jina-embeddings-v2-small-en"


class OffTopicJina(HuggingFace):
    """Wrapper for off-topic detection model from govtech.

    For more information, please see the model card:

    - [govtech/jina-embeddings-v2-small-en-off-topic](https://huggingface.co/govtech/jina-embeddings-v2-small-en-off-topic).
    """

    SUPPORTED_MODELS: ClassVar = ["mozilla-ai/jina-embeddings-v2-small-en-off-topic"]

    def _load_model(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(BASEMODEL)  # type: ignore[no-untyped-call]
        base_model = AutoModel.from_pretrained(BASEMODEL)
        self.model = CrossEncoderWithSharedBase.from_pretrained(self.model_id, base_model=base_model)

    def _pre_processing(
        self, input_text: str, comparison_text: str | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        warnings.warn("Truncating input text to a max length of 1024 tokens.", stacklevel=2)
        inputs1 = self.tokenizer(
            input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=1024
        )
        inputs2 = self.tokenizer(
            comparison_text, return_tensors="pt", truncation=True, padding="max_length", max_length=1024
        )
        input_ids1 = inputs1["input_ids"]  # .to(device)
        attention_mask1 = inputs1["attention_mask"]  # .to(device)
        input_ids2 = inputs2["input_ids"]  # .to(device)
        attention_mask2 = inputs2["attention_mask"]  # .to(device)
        return input_ids1, attention_mask1, input_ids2, attention_mask2

    def _inference(
        self,
        model_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor],
    ) -> Any:
        if len(model_inputs) != 4:
            msg = "Expected model_inputs to be a tuple of (input_ids1, attention_mask1, input_ids2, attention_mask2)."
            raise ValueError(msg)
        input_ids1, attention_mask1, input_ids2, attention_mask2 = model_inputs
        with torch.no_grad():
            return self.model(
                input_ids1=input_ids1,
                attention_mask1=attention_mask1,
                input_ids2=input_ids2,
                attention_mask2=attention_mask2,
            )

    def _post_processing(self, model_outputs: Any) -> GuardrailOutput:
        probabilities = torch.softmax(model_outputs, dim=1)
        predicted_label = torch.argmax(probabilities, dim=1).item()
        explanatory_probs = probabilities.cpu().numpy().tolist()[0]
        probs_dict = {"on-topic": explanatory_probs[0], "off-topic": explanatory_probs[1]}

        return GuardrailOutput(
            valid=predicted_label != 1,  # Assuming label '1' indicates off-topic
            explanation=probs_dict,
        )
