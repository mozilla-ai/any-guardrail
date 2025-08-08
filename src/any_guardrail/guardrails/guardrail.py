from abc import ABC, abstractmethod
from typing import Any
from any_guardrail.types import GuardrailOutput, GuardrailModel


class Guardrail(ABC):
    def __init__(self, model_id: str):
        self.model_id = model_id

    @abstractmethod
    def validate(self, *args: Any, **kwargs: Any) -> GuardrailOutput:
        """
        Abstract method for validating some input. Each subclass implements its own signature.
        """
        raise NotImplementedError("Each subclass will create their own method.")

    @abstractmethod
    def _model_instantiation(self) -> GuardrailModel:
        raise NotImplementedError("Each subclass will create their own method.")
