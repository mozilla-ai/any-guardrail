from typing import Any

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    MISSING_PACKAGES_ERROR = None

except ImportError as e:
    MISSING_PACKAGES_ERROR = e

from any_guardrail.providers.base import Provider
from any_guardrail.types import (
    AnyDict,
    GuardrailInferenceOutput,
    GuardrailPreprocessOutput,
)


class HuggingFaceProvider(Provider[AnyDict, AnyDict]):
    """Standard HuggingFace provider for sequence classification models.

    Handles model loading via transformers, tokenization-based preprocessing,
    and PyTorch inference with torch.no_grad().

    Args:
        model_class: The transformers model class to use for loading. Defaults to
            AutoModelForSequenceClassification.
        tokenizer_class: The transformers tokenizer class to use. Defaults to AutoTokenizer.
        tokenizer_id: Override the tokenizer model ID (defaults to the same as model_id).
        trust_remote_code: Whether to trust remote code when loading models.
        device: The torch device to load the model on (e.g., ``"cpu"``, ``"cuda"``,
            ``"cuda:0"``, ``"mps"``). When ``None``, the model stays on the device
            chosen by ``transformers`` (typically CPU). Tokenized inputs are moved
            to this device automatically before inference.

    """

    def __init__(
        self,
        model_class: type | None = None,
        tokenizer_class: type | None = None,
        tokenizer_id: str | None = None,
        trust_remote_code: bool = False,
        device: str | None = None,
    ) -> None:
        """Initialize the HuggingFace provider."""
        if MISSING_PACKAGES_ERROR is not None:
            msg = "Missing packages for HuggingFace provider. You can try `pip install 'any-guardrail[huggingface]'`"
            raise ImportError(msg) from MISSING_PACKAGES_ERROR

        self.model_class = model_class or AutoModelForSequenceClassification
        self.tokenizer_class = tokenizer_class or AutoTokenizer
        self.tokenizer_id = tokenizer_id
        self.trust_remote_code = trust_remote_code
        self.device = device
        self.model: Any = None
        self.tokenizer: Any = None

    def load_model(self, model_id: str, **kwargs: Any) -> None:
        """Load model and tokenizer from HuggingFace.

        Args:
            model_id: The HuggingFace model identifier.
            **kwargs: Additional keyword arguments passed to from_pretrained.

        """
        load_kwargs: AnyDict = {}
        if self.trust_remote_code:
            load_kwargs["trust_remote_code"] = True

        self.model = self.model_class.from_pretrained(model_id, **load_kwargs, **kwargs)  # type: ignore[attr-defined]
        if self.device is not None:
            self.model = self.model.to(self.device)
        tokenizer_id = self.tokenizer_id or model_id
        self.tokenizer = self.tokenizer_class.from_pretrained(tokenizer_id, **load_kwargs)  # type: ignore[attr-defined]

    def pre_process(self, input_text: str | list[str], **kwargs: Any) -> GuardrailPreprocessOutput[AnyDict]:
        """Tokenize input text for model consumption.

        Args:
            input_text: A single text or a list of texts to preprocess. When a list
                is supplied, ``padding=True`` is added automatically (unless caller
                already specified a ``padding`` argument) so the resulting tensors
                can be stacked into a batch.
            **kwargs: Additional keyword arguments passed to the tokenizer.

        Returns:
            GuardrailPreprocessOutput wrapping the tokenized input.

        """
        if isinstance(input_text, list) and "padding" not in kwargs:
            kwargs["padding"] = True
        tokenized = self.tokenizer(input_text, return_tensors="pt", **kwargs)
        if self.device is not None:
            tokenized = tokenized.to(self.device)
        return GuardrailPreprocessOutput(data=tokenized)

    def infer(self, model_inputs: GuardrailPreprocessOutput[AnyDict]) -> GuardrailInferenceOutput[AnyDict]:
        """Run model inference on preprocessed inputs.

        Args:
            model_inputs: The wrapped preprocessing output.

        Returns:
            GuardrailInferenceOutput wrapping the model output.

        """
        with torch.no_grad():
            output = self.model(**model_inputs.data)
        return GuardrailInferenceOutput(data=output)
