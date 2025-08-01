from dataclasses import dataclass
from typing import Optional, Dict
from transformers import Pipeline, PreTrainedModel # type: ignore[attr-defined]
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from flow_judge import FlowJudge

@dataclass
class ClassificationOutput:
    unsafe: Optional[bool] = None
    explanation: Optional[str | Dict[str, bool]] = None
    score: Optional[float | int] = None

@dataclass
class GuardrailModel:
    model: Optional[Pipeline | PreTrainedModel | FlowJudge] = None
    tokenizer: Optional[PreTrainedTokenizerBase] = None

