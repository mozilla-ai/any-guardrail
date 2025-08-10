from abc import ABC, abstractmethod
from typing import Any
from any_guardrail.types import GuardrailOutput
from enum import Enum


class Guardrail(ABC):
    SUPPORTED_MODELS: list[str]

    def __init__(self, model_id: str):
        self.model_id = model_id
        self._validate_model_id(model_id)
        self._load_model()

    def _validate_model_id(self, model_id: str) -> None:
        if model_id not in self.SUPPORTED_MODELS:
            raise ValueError(f"Only supports {self.SUPPORTED_MODELS}. Please use this path to instantiate model.")

    @abstractmethod
    def validate(self, *args: Any, **kwargs: Any) -> GuardrailOutput:
        """
        Abstract method for validating some input. Each subclass implements its own signature.
        """
        raise NotImplementedError("Each subclass will create their own method.")

    @abstractmethod
    def _load_model(self) -> None:
        raise NotImplementedError("Each subclass will create their own method.")


class GuardrailName(str, Enum):
    """String enum for supported guardrails"""

    DEEPSET = "deepset"
    DUOGUARD = "duo_guard"
    FLOWJUDGE = "flowjudge"
    GLIDER = "glider"
    HARMGUARD = "harm_guard"
    INJECGUARD = "injec_guard"
    JASPER = "jasper"
    PANGOLIN = "pangolin"
    PROTECTAI = "protectai"
    SENTINEL = "sentinel"
    SHIELD_GEMMA = "shield_gemma"
