from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, ClassVar

from any_guardrail.types import GuardrailOutput


class Guardrail(ABC):
    """Base class for all guardrails."""

    SUPPORTED_MODELS: ClassVar[list[str]] = []

    def __init__(self, model_id: str):
        """Initialize the guardrail with a model ID."""
        self.model_id = model_id
        self._validate_model_id(model_id)
        self._load_model()

    def _validate_model_id(self, model_id: str) -> None:
        if model_id not in self.SUPPORTED_MODELS:
            msg = f"Only supports {self.SUPPORTED_MODELS}. Please use this path to instantiate model."
            raise ValueError(msg)

    @abstractmethod
    def validate(self, *args: Any, **kwargs: Any) -> GuardrailOutput:
        """Abstract method for validating some input. Each subclass implements its own signature."""
        msg = "Each subclass will create their own method."
        raise NotImplementedError(msg)

    @abstractmethod
    def _load_model(self) -> None:
        msg = "Each subclass will create their own method."
        raise NotImplementedError(msg)


class GuardrailName(str, Enum):
    """String enum for supported guardrails."""

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
