from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, ClassVar

from pydantic import BaseModel


class GuardrailName(str, Enum):
    """String enum for supported guardrails."""

    ANYLLM = "any_llm"
    DEEPSET = "deepset"
    DUOGUARD = "duo_guard"
    FLOWJUDGE = "flowjudge"
    GLIDER = "glider"
    HARMGUARD = "harm_guard"
    INJECGUARD = "injec_guard"
    JASPER = "jasper"
    OFFTOPIC = "off_topic"
    PANGOLIN = "pangolin"
    PROTECTAI = "protectai"
    SENTINEL = "sentinel"
    SHIELD_GEMMA = "shield_gemma"


class GuardrailOutput(BaseModel):
    """Represents the output of a guardrail evaluation."""

    valid: bool | None = None
    """Indicates if the output should be considered valid."""

    explanation: str | dict[str, bool] | dict[str, float] | None = None
    """Provides an explanation for the guardrail evaluation result."""

    score: float | int | None = None
    """Represents the score assigned to the output by the guardrail."""


class Guardrail(ABC):
    """Base class for all guardrails."""

    SUPPORTED_MODELS: ClassVar[list[str]] = []

    @abstractmethod
    def validate(self, *args: Any, **kwargs: Any) -> GuardrailOutput:
        """Abstract method for validating some input. Each subclass implements its own signature."""
        msg = "Each subclass will create their own method."
        raise NotImplementedError(msg)
