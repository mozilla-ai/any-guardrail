from typing import Any

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    MISSING_PACKAGES_ERROR = None

except ImportError as e:
    MISSING_PACKAGES_ERROR = e

from any_guardrail.providers.base import Provider
from any_guardrail.types import (
    GuardrailInferenceOutput,
    GuardrailPreprocessOutput,
)

# Type alias for standard HuggingFace dict types
HFDict = dict[str, Any]


class HuggingFaceProvider(Provider[HFDict, HFDict]):
    """Standard HuggingFace provider for sequence classification models.

    Handles model loading via transformers, tokenization-based preprocessing,
    and PyTorch inference with torch.no_grad().

    Args:
        model_class: The transformers model class to use for loading. Defaults to
            AutoModelForSequenceClassification.
        tokenizer_class: The transformers tokenizer class to use. Defaults to AutoTokenizer.
        tokenizer_id: Override the tokenizer model ID (defaults to the same as model_id).
        trust_remote_code: Whether to trust remote code when loading models.

    """

    def __init__(
        self,
        model_class: type | None = None,
        tokenizer_class: type | None = None,
        tokenizer_id: str | None = None,
        trust_remote_code: bool = False,
    ) -> None:
        if MISSING_PACKAGES_ERROR is not None:
            msg = "Missing packages for HuggingFace provider. You can try `pip install 'any-guardrail[huggingface]'`"
            raise ImportError(msg) from MISSING_PACKAGES_ERROR

        self.model_class = model_class or AutoModelForSequenceClassification
        self.tokenizer_class = tokenizer_class or AutoTokenizer
        self.tokenizer_id = tokenizer_id
        self.trust_remote_code = trust_remote_code
        self.model: Any = None
        self.tokenizer: Any = None

    def load_model(self, model_id: str, **kwargs: Any) -> None:
        """Load model and tokenizer from HuggingFace.

        Args:
            model_id: The HuggingFace model identifier.
            **kwargs: Additional keyword arguments passed to from_pretrained.

        """
        load_kwargs: dict[str, Any] = {}
        if self.trust_remote_code:
            load_kwargs["trust_remote_code"] = True

        self.model = self.model_class.from_pretrained(model_id, **load_kwargs, **kwargs)
        tokenizer_id = self.tokenizer_id or model_id
        self.tokenizer = self.tokenizer_class.from_pretrained(tokenizer_id, **load_kwargs)

    def pre_process(self, input_text: str, **kwargs: Any) -> GuardrailPreprocessOutput[HFDict]:
        """Tokenize input text for model consumption.

        Args:
            input_text: The text to preprocess.
            **kwargs: Additional keyword arguments passed to the tokenizer.

        Returns:
            GuardrailPreprocessOutput wrapping the tokenized input.

        """
        tokenized = self.tokenizer(input_text, return_tensors="pt", **kwargs)
        return GuardrailPreprocessOutput(data=tokenized)

    def infer(self, model_inputs: GuardrailPreprocessOutput[HFDict]) -> GuardrailInferenceOutput[HFDict]:
        """Run model inference on preprocessed inputs.

        Args:
            model_inputs: The wrapped preprocessing output.

        Returns:
            GuardrailInferenceOutput wrapping the model output.

        """
        with torch.no_grad():
            output = self.model(**model_inputs.data)
        return GuardrailInferenceOutput(data=output)
