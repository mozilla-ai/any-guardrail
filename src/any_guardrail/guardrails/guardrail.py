from abc import ABC, abstractmethod
from typing import Any


class Guardrail(ABC):
    def __init__(self, modelpath: str):
        self.modelpath = modelpath

    @abstractmethod
    def classify(self) -> Any:
        raise NotImplementedError("Each subclass will creat their own method.")

    @abstractmethod
    def _model_instantiation(self) -> Any:
        raise NotImplementedError("Each subclass will creat their own method.")
