from pydantic import BaseModel


class GuardrailOutput(BaseModel):
    """Represents the output of a guardrail evaluation."""

    unsafe: bool | None = None
    """Indicates if the output is considered unsafe."""

    explanation: str | dict[str, bool] | dict[str, float] | None = None
    """Provides an explanation for the guardrail evaluation result."""

    score: float | int | None = None
    """Represents the score assigned to the output by the guardrail."""
