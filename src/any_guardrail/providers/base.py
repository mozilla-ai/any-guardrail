from abc import ABC, abstractmethod
from typing import Any, Generic

from any_guardrail.types import (
    AnyDict,
    GuardrailInferenceOutput,
    GuardrailPreprocessOutput,
    InferenceT,
    PreprocessT,
)


class Provider(ABC, Generic[PreprocessT, InferenceT]):
    """Base class for execution providers.

    A provider handles model loading, preprocessing, and inference.
    Post-processing remains the guardrail's responsibility.

    Type Parameters:
        PreprocessT: The type of data produced by preprocessing.
        InferenceT: The type of data produced by inference.

    """

    @abstractmethod
    def load_model(self, model_id: str, **kwargs: Any) -> None:
        """Load the model and any required components."""
        ...

    @abstractmethod
    def pre_process(self, *args: Any, **kwargs: Any) -> GuardrailPreprocessOutput[PreprocessT]:
        """Transform raw input into model-ready format."""
        ...

    @abstractmethod
    def infer(self, model_inputs: GuardrailPreprocessOutput[PreprocessT]) -> GuardrailInferenceOutput[InferenceT]:
        """Run inference on preprocessed inputs."""
        ...


# Standard provider type alias
StandardProvider = Provider[AnyDict, AnyDict]
"""Type alias for providers using standard AnyDict inputs/outputs."""
