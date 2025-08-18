from typing import Any, ClassVar

import torch

from any_guardrail.guardrail import Guardrail
from any_guardrail.guardrails.huggingface import HuggingFace
from any_guardrail.types import GuardrailOutput
from any_guardrail.guardrails.off_topic.off_topic_stsb import OffTopicStsb
from any_guardrail.guardrails.off_topic.off_topic_jina import OffTopicJina  


class OffTopic(HuggingFace):
    """Abstract base class for the Off Topic models."""

    SUPPORTED_MODELS: ClassVar = ["mozilla-ai/jina-embeddings-v2-small-en-off-topic", 
                                  "mozilla-ai/stsb-roberta-base-off-topic"]
    
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        if self.model_id == self.SUPPORTED_MODELS[0]:
            self.implementation = OffTopicJina()
        elif self.model_id == self.SUPPORTED_MODELS[1]:
            self.implementation = OffTopicStsb()  # type: ignore [assignment]
        else:
            raise ValueError(f"Unsupported model_id: {self.model_id}")
        super().__init__()

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
        model_inputs = self.implementation._pre_processing(input_text, comparison_text)
        model_outputs = self.implementation._inference(model_inputs)
        return self.implementation._post_processing(model_outputs)
    
    def _load_model(self) -> None:
        self.implementation._load_model()
