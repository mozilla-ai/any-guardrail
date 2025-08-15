import warnings
from typing import Any, ClassVar

import torch
from transformers import AutoModel, AutoTokenizer

from any_guardrail.guardrails.huggingface import HuggingFace
from any_guardrail.guardrails.off_topic_jina.models.cross_encoder_shared import CrossEncoderWithSharedBase
from any_guardrail.types import GuardrailOutput

BASEMODEL = "jinaai/jina-embeddings-v2-small-en"


class OffTopicJina(HuggingFace):
    """Wrapper for off-topic detection model from govtech.

    For more information, please see the model card:

    - [govtech/jina-embeddings-v2-small-en-off-topic](https://huggingface.co/govtech/jina-embeddings-v2-small-en-off-topic).
    """

    SUPPORTED_MODELS: ClassVar = ["mozilla-ai/jina-embeddings-v2-small-en-off-topic"]

    def validate(self, input_text: str, comparison_text: str = "") -> GuardrailOutput:
        """Compare two texts to see if they are relevant to each other.

        Args:
            input_text: the original text you want to compare against.
            comparison_text: the text you want to compare to.

        Returns:
            Unsafe means off topic, safe means on topic. Will also provide probabilities of each.

        """
        msg = "Must provide a text to compare to."
        if len(comparison_text) == 0:
            raise ValueError(msg)
        model_inputs = self._pre_processing(input_text, comparison_text)
        model_outputs = self._inference(model_inputs)
        return self._post_processing(model_outputs)

    def _load_model(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(BASEMODEL)  # type: ignore[no-untyped-call]
        base_model = AutoModel.from_pretrained(BASEMODEL)
        self.model = CrossEncoderWithSharedBase.from_pretrained(self.model_id, base_model=base_model)

    def _pre_processing(
        self, input_text: str, comparison_text: str = ""
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
        model_inputs: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> Any:
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
            unsafe=predicted_label == 1,  # Assuming label '1' indicates off-topic
            explanation=probs_dict,
        )
