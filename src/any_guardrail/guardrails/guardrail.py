from abc import ABC, abstractmethod
from any_guardrail.utils.custom_types import ClassificationOutput, GuardrailModel


class Guardrail(ABC):
    def __init__(self, model_identifier: str):
        self.model_identifier = model_identifier

    @abstractmethod
    def safety_review(self) -> ClassificationOutput:
        raise NotImplementedError("Each subclass will creat their own method.")

    @abstractmethod
    def _model_instantiation(self) -> GuardrailModel:
        raise NotImplementedError("Each subclass will creat their own method.")
