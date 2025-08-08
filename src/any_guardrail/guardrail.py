from abc import ABC, abstractmethod
from typing import Any
from any_guardrail.types import ClassificationOutput, GuardrailModel
from enum import Enum


class Guardrail(ABC):
    def __init__(self, model_id: str):
        self.model_id = model_id

    @abstractmethod
    def safety_review(self, *args: Any, **kwargs: Any) -> ClassificationOutput:
        """
        Abstract method for safety review. Each subclass implements its own signature.
        """
        raise NotImplementedError("Each subclass will create their own method.")

    @abstractmethod
    def _model_instantiation(self) -> GuardrailModel:
        raise NotImplementedError("Each subclass will create their own method.")


class GuardrailName(str, Enum):
    """String enum for supported guardrails"""

    DEEPSET = "deepset"
    DUOGUARD = "duoguard"
    FLOWJUDGE = "flowjudge"
    GLIDER = "glider"
    HARMGUARD = "harmguard"
    INJECGUARD = "injecguard"
    JASPER = "jasper"
    PANGOLIN = "pangolin"
    PROTECTAI = "protectai"
    SENTINEL = "sentinel"
    SHIELD_GEMMA = "shield_gemma"
