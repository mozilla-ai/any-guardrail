from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class GuardrailOutput:
    unsafe: Optional[bool] = None
    explanation: Optional[str | Dict[str, bool]] = None
    score: Optional[float | int] = None
