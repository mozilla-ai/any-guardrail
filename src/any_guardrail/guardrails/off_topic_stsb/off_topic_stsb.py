from typing import Any, ClassVar
import warnings

import torch
from transformers import AutoModel, AutoTokenizer

from any_guardrail.guardrails.huggingface import HuggingFace
from any_guardrail.types import GuardrailOutput
from any_guardrail.guardrails.off_topic_stsb.models.cross_encoder_mlp import CrossEncoderWithMLP

BASEMODEL = "cross-encoder/stsb-roberta-base"


class OffTopicStsb(HuggingFace):
    """Wrapper for off-topic detection model from govtech.

    For more information, please see the model card:

    - [govtech/stsb-roberta-base-off-topic model](https://huggingface.co/govtech/stsb-roberta-base-off-topic).
    """

    SUPPORTED_MODELS: ClassVar = ["mozilla-ai/stsb-roberta-base-off-topic",]

    def validate(self, input_text: str, comparison_text: str = "") -> GuardrailOutput:
        """Compare two texts to see if they are relevant to each other.
        
        Args:
            input_text: the original text you want to compare against.
            comparison_text: the text you want to compare to.

        returns:
            Unsafe means off topic, safe means on topic. Will also provide probabilities of each.
        """
        if len(comparison_text) == 0:
            raise ValueError("Must provide a text to compare to.")
        model_inputs = self._pre_processing(input_text, comparison_text)
        model_outputs = self._inference(model_inputs)
        return self._post_processing(model_outputs)

    def _load_model(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(BASEMODEL)  # type: ignore[no-untyped-call]
        base_model = AutoModel.from_pretrained(BASEMODEL)
        self.model = CrossEncoderWithMLP.from_pretrained(self.model_id, base_model=base_model)

    def _pre_processing(self, input_text: str, comparison_text: str = "") -> tuple[torch.Tensor, torch.Tensor]:
        warnings.warn("Truncating text to a maximum length of 514 tokens.")
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

    def _inference(self, model_inputs: tuple[torch.Tensor, torch.Tensor]) -> Any:
        input_ids, attention_mask = model_inputs
        with torch.no_grad():
            return self.model(input_ids=input_ids, attention_mask=attention_mask)

    def _post_processing(self, model_outputs: Any) -> GuardrailOutput:
        probabilities = torch.softmax(model_outputs, dim=1)
        predicted_label = torch.argmax(probabilities, dim=1).item()
        explanatory_probs = probabilities.cpu().numpy().tolist()[0]
        probs_dict = {"on-topic": explanatory_probs[0], "off-topic": explanatory_probs[1]}

        return GuardrailOutput(
            unsafe=predicted_label == 1,  # Assuming label '1' indicates off-topic
            explanation=probs_dict,
        )
