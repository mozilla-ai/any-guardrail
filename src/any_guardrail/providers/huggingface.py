from typing import Any, ClassVar

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
        trust_remote_code: Whether to trust remote code when loading models. Forwarded
            to both the model and tokenizer ``from_pretrained`` calls.
        device: The torch device to load the model on (e.g., ``"cpu"``, ``"cuda"``,
            ``"cuda:0"``, ``"mps"``). When ``None``, the model stays on the device
            chosen by ``transformers`` (typically CPU). Tokenized inputs are moved
            to this device automatically before inference.
        torch_dtype: The torch dtype to load the model with (e.g. ``torch.float16``,
            ``torch.bfloat16``). Forwarded to the model's ``from_pretrained``.
        cache_dir: Filesystem path used to cache downloaded model and tokenizer
            files. Forwarded to ``from_pretrained`` for both model and tokenizer.
        revision: The git revision (branch, tag, or commit) to load. Forwarded to
            ``from_pretrained`` for both model and tokenizer.
        model_kwargs: Additional keyword arguments to forward to the model's
            ``from_pretrained``. Use this to set parameters not surfaced above
            (e.g. ``device_map``, ``low_cpu_mem_usage``, ``attn_implementation``).
        tokenizer_kwargs: Additional keyword arguments forwarded to the tokenizer
            on every ``pre_process`` call (e.g. ``max_length``, ``truncation``,
            ``padding``). Values can be overridden per-call.

    """

    # Keys that have dedicated constructor parameters and would create
    # model/tokenizer mismatches if also passed via ``model_kwargs``.
    _RESERVED_MODEL_KWARGS: ClassVar[frozenset[str]] = frozenset(
        {"trust_remote_code", "cache_dir", "revision", "torch_dtype"}
    )

    def __init__(
        self,
        model_class: type | None = None,
        tokenizer_class: type | None = None,
        tokenizer_id: str | None = None,
        trust_remote_code: bool = False,
        device: str | None = None,
        torch_dtype: Any | None = None,
        cache_dir: str | None = None,
        revision: str | None = None,
        model_kwargs: AnyDict | None = None,
        tokenizer_kwargs: AnyDict | None = None,
    ) -> None:
        """Initialize the HuggingFace provider."""
        if MISSING_PACKAGES_ERROR is not None:
            msg = "Missing packages for HuggingFace provider. You can try `pip install 'any-guardrail[huggingface]'`"
            raise ImportError(msg) from MISSING_PACKAGES_ERROR

        if model_kwargs:
            reserved_used = self._RESERVED_MODEL_KWARGS & model_kwargs.keys()
            if reserved_used:
                msg = (
                    f"model_kwargs cannot contain reserved keys {sorted(reserved_used)}; "
                    f"use the dedicated constructor parameters instead so the model and "
                    f"tokenizer stay in sync."
                )
                raise ValueError(msg)

        self.model_class = model_class or AutoModelForSequenceClassification
        self.tokenizer_class = tokenizer_class or AutoTokenizer
        self.tokenizer_id = tokenizer_id
        self.trust_remote_code = trust_remote_code
        self.device = device
        self.torch_dtype = torch_dtype
        self.cache_dir = cache_dir
        self.revision = revision
        self.model_kwargs: AnyDict = dict(model_kwargs) if model_kwargs else {}
        self.tokenizer_kwargs: AnyDict = dict(tokenizer_kwargs) if tokenizer_kwargs else {}
        self.model: Any = None
        self.tokenizer: Any = None

    def _build_from_pretrained_kwargs(self) -> AnyDict:
        """Collect the explicit, surfaced parameters into a kwargs dict for ``from_pretrained``."""
        kwargs: AnyDict = {}
        if self.trust_remote_code:
            kwargs["trust_remote_code"] = True
        if self.cache_dir is not None:
            kwargs["cache_dir"] = self.cache_dir
        if self.revision is not None:
            kwargs["revision"] = self.revision
        return kwargs

    def load_model(self, model_id: str, **kwargs: Any) -> None:
        """Load model and tokenizer from HuggingFace.

        Surfaced provider-level parameters (``trust_remote_code``, ``torch_dtype``,
        ``cache_dir``, ``revision``, ``model_kwargs``) are merged with the kwargs
        passed here; call-site ``kwargs`` win on conflict.

        Args:
            model_id: The HuggingFace model identifier.
            **kwargs: Additional keyword arguments passed to the model's
                ``from_pretrained`` (override provider-level defaults).

        """
        common_kwargs = self._build_from_pretrained_kwargs()
        model_load_kwargs: AnyDict = {**common_kwargs, **self.model_kwargs}
        if self.torch_dtype is not None:
            model_load_kwargs["torch_dtype"] = self.torch_dtype
        model_load_kwargs.update(kwargs)
        self.model = self.model_class.from_pretrained(model_id, **model_load_kwargs)  # type: ignore[attr-defined]
        if self.device is not None:
            self.model = self.model.to(self.device)
        tokenizer_id = self.tokenizer_id or model_id
        self.tokenizer = self.tokenizer_class.from_pretrained(tokenizer_id, **common_kwargs)  # type: ignore[attr-defined]

    def pre_process(self, input_text: str, **kwargs: Any) -> GuardrailPreprocessOutput[AnyDict]:
        """Tokenize input text for model consumption.

        Provider-level ``tokenizer_kwargs`` (e.g. ``max_length``, ``truncation``,
        ``padding``) are applied first; per-call ``kwargs`` override them.
        ``return_tensors`` defaults to ``"pt"`` but can be overridden via either
        ``tokenizer_kwargs`` or per-call ``kwargs``.

        Args:
            input_text: The text to preprocess.
            **kwargs: Additional keyword arguments passed to the tokenizer.

        Returns:
            GuardrailPreprocessOutput wrapping the tokenized input.

        """
        call_kwargs: AnyDict = {**self.tokenizer_kwargs, **kwargs}
        call_kwargs.setdefault("return_tensors", "pt")
        tokenized = self.tokenizer(input_text, **call_kwargs)
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
