from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, ClassVar

from any_guardrail.types import GuardrailOutput


class Guardrail(ABC):
    """Base class for all guardrails."""

    SUPPORTED_MODELS: ClassVar[list[str]] = []

    @abstractmethod
    def validate(self, *args: Any, **kwargs: Any) -> GuardrailOutput:
        """Abstract method for validating some input. Each subclass implements its own signature."""
        msg = "Each subclass will create their own method."
        raise NotImplementedError(msg)


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
    OFFTOPICSTSB = "off_topic_stsb"
    OFFTOPICJINA = "off_topic_jina"
    PANGOLIN = "pangolin"
    PROTECTAI = "protectai"
    SENTINEL = "sentinel"
    SHIELD_GEMMA = "shield_gemma"
