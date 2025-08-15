from typing import Any, ClassVar

import torch
from transformers import AutoModel, AutoTokenizer

from any_guardrail.guardrails.huggingface import HuggingFace
from any_guardrail.types import GuardrailOutput


class OffTopic(HuggingFace):
    """Wrapper for off-topic detection model from govtech.

    For more information, please see the model card:

    - [govtech/stsb-roberta-base-off-topic model](https://huggingface.co/govtech/stsb-roberta-base-off-topic).
    - [govtech/jina-embeddings-v2-small-en-off-topic](https://huggingface.co/govtech/jina-embeddings-v2-small-en-off-topic).
    """

    SUPPORTED_MODELS: ClassVar = [
        "mozilla-ai/stsb-roberta-base-off-topic",
        "mozilla-ai/jina-embeddings-v2-small-en-off-topic",
    ]

    def validate(self, input_text: str, comparison_text: str = "") -> GuardrailOutput:
        """Validate whether the input text is safe or not."""
        model_inputs = self._pre_processing(input_text, comparison_text)
        model_outputs = self._inference(model_inputs)
        return self._post_processing(model_outputs)

    def _load_model(self) -> None:
        if self.model_id == "mozilla-ai/stsb-roberta-base-off-topic":
            from any_guardrail.guardrails.off_topic.models.cross_encoder_mlp import CrossEncoderWithMLP

            self.tokenizer = AutoTokenizer.from_pretrained("cross-encoder/stsb-roberta-base")  # type: ignore[no-untyped-call]
            base_model = AutoModel.from_pretrained("cross-encoder/stsb-roberta-base")
            self.model = CrossEncoderWithMLP.from_pretrained(self.model_id, base_model=base_model)
        else:
            from any_guardrail.guardrails.off_topic.models.cross_encoder_shared import CrossEncoderWithSharedBase

            self.tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v2-small-en")  # type: ignore[no-untyped-call]
            base_model = AutoModel.from_pretrained("jinaai/jina-embeddings-v2-small-en")
            self.model = CrossEncoderWithSharedBase.from_pretrained(self.model_id, base_model=base_model)

    def _pre_processing(self, input_text: str, comparison_text: str = "") -> tuple[torch.Tensor, ...]:
        if self.model_id == "mozilla-ai/stsb-roberta-base-off-topic":
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
            return input_ids, attention_mask
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

    def _inference(self, model_inputs: tuple[torch.Tensor, ...]) -> Any:
        if self.model_id == "mozilla-ai/stsb-roberta-base-off-topic":
            input_ids, attention_mask = model_inputs
            with torch.no_grad():
                return self.model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            input_ids1, attention_mask1, input_ids2, attention_mask2 = model_inputs
            assert input_ids2 is not None, "Comparison text inputs must be provided for the shared base model."
            assert attention_mask2 is not None, "Comparison text inputs must be provided for the shared base model."
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
            unsafe=predicted_label == 1,  # Assuming label '1' indicates off-topic
            explanation=probs_dict,
        )
