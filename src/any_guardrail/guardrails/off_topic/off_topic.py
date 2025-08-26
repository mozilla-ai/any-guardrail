from typing import Any, ClassVar

from any_guardrail.base import GuardrailOutput
from any_guardrail.guardrails.huggingface import HuggingFace
from any_guardrail.guardrails.off_topic.off_topic_jina import OffTopicJina
from any_guardrail.guardrails.off_topic.off_topic_stsb import OffTopicStsb


class OffTopic(HuggingFace):
    """Abstract base class for the Off Topic models.

    For more information about the implementations about either off topic model, please see the below model cards:

    - [govtech/stsb-roberta-base-off-topic model](https://huggingface.co/govtech/stsb-roberta-base-off-topic).
    - [govtech/jina-embeddings-v2-small-en-off-topic](https://huggingface.co/govtech/jina-embeddings-v2-small-en-off-topic).
    """

    SUPPORTED_MODELS: ClassVar = [
        "mozilla-ai/jina-embeddings-v2-small-en-off-topic",
        "mozilla-ai/stsb-roberta-base-off-topic",
    ]

    implementation: OffTopicJina | OffTopicStsb

    def __init__(self, model_id: str | None = None) -> None:
        """Off Topic model based on one of two implementations decided by model ID."""
        super().__init__(model_id)
        if self.model_id == self.SUPPORTED_MODELS[0]:
            self.implementation = OffTopicJina()
        elif self.model_id == self.SUPPORTED_MODELS[1]:
            self.implementation = OffTopicStsb()
        else:
            msg = f"Unsupported model_id: {self.model_id}"
            raise ValueError(msg)
        super().__init__()

    def validate(self, input_text: str, comparison_text: str | None = None) -> GuardrailOutput:
        """Compare two texts to see if they are relevant to each other.

        Args:
            input_text: the original text you want to compare against.
            comparison_text: the text you want to compare to.

        Returns:
            valid=False means off topic, valid=True  means on topic. Will also provide probabilities of each.

        """
        msg = "Must provide a text to compare to."
        if comparison_text:
            raise ValueError(msg)
        model_inputs = self.implementation._pre_processing(input_text, comparison_text)
        model_outputs = self.implementation._inference(model_inputs)
        return self._post_processing(model_outputs)

    def _load_model(self) -> None:
        self.implementation._load_model()

    def _post_processing(self, model_outputs: Any) -> GuardrailOutput:
        return self.implementation._post_processing(model_outputs)
