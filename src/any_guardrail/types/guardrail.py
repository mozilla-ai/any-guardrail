from dataclasses import dataclass


@dataclass
class GuardrailOutput:
    """Represents the output of a guardrail evaluation."""

    unsafe: bool | None = None
    """Indicates if the output is considered unsafe."""

    explanation: str | dict[str, bool] | None = None
    """Provides an explanation for the guardrail evaluation result."""

    score: float | int | None = None
    """Represents the score assigned to the output by the guardrail."""
