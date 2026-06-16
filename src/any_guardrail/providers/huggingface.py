from typing import Any, ClassVar

try:
    import numpy as np
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


def _softmax(logits: "np.ndarray[Any, Any]") -> "np.ndarray[Any, Any]":
    maxes = np.max(logits, axis=-1, keepdims=True)
    shifted_exp = np.exp(logits - maxes)
    result: np.ndarray[Any, Any] = shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)
    return result


def _sigmoid(logits: "np.ndarray[Any, Any]") -> "np.ndarray[Any, Any]":
    result: np.ndarray[Any, Any] = 1.0 / (1.0 + np.exp(-logits))
    return result


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
        multi_label: bool = False,
    ) -> None:
        """Initialize the HuggingFace provider.

        ``multi_label=True`` switches the activation in ``infer()`` from softmax
        to sigmoid, so ``scores`` carries per-class probabilities for models
        whose categories are not mutually exclusive (e.g. DuoGuard).
        """
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
        self.multi_label = multi_label
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
                ``from_pretrained`` (override provider-level defaults). Two reserved
                names are popped before forwarding:

                - ``model_class``: override the model class used for this load.
                  Lets a decoder-LM guardrail (e.g. ShieldGemma) enforce
                  ``AutoModelForCausalLM`` even when the caller supplied a
                  default-constructed ``HuggingFaceProvider`` configured for
                  sequence classification. ``self.model_class`` is not mutated.
                - ``tokenizer_class``: same idea for the tokenizer class (e.g.
                  ``AutoProcessor`` for Llama Guard 4).

        """
        model_class = kwargs.pop("model_class", None) or self.model_class
        tokenizer_class = kwargs.pop("tokenizer_class", None) or self.tokenizer_class
        common_kwargs = self._build_from_pretrained_kwargs()
        model_load_kwargs: AnyDict = {**common_kwargs, **self.model_kwargs}
        if self.torch_dtype is not None:
            model_load_kwargs["torch_dtype"] = self.torch_dtype
        model_load_kwargs.update(kwargs)
        self.model = model_class.from_pretrained(model_id, **model_load_kwargs)  # type: ignore[union-attr]
        if self.device is not None:
            self.model = self.model.to(self.device)
        tokenizer_id = self.tokenizer_id or model_id
        self.tokenizer = tokenizer_class.from_pretrained(tokenizer_id, **common_kwargs)  # type: ignore[union-attr]

    def pre_process(self, input_text: str | list[str], **kwargs: Any) -> GuardrailPreprocessOutput[AnyDict]:
        """Tokenize input text for model consumption.

        Provider-level ``tokenizer_kwargs`` (e.g. ``max_length``, ``truncation``,
        ``padding``) are applied first; per-call ``kwargs`` override them.
        ``return_tensors`` defaults to ``"pt"`` but can be overridden via either
        ``tokenizer_kwargs`` or per-call ``kwargs``.

        Args:
            input_text: A single text or a list of texts to preprocess. When a list
                is supplied, ``padding=True`` is added automatically (unless caller
                already specified a ``padding`` argument) so the resulting tensors
                can be stacked into a batch.
            **kwargs: Additional keyword arguments passed to the tokenizer.

        Returns:
            GuardrailPreprocessOutput wrapping the tokenized input.

        """
        call_kwargs: AnyDict = {**self.tokenizer_kwargs, **kwargs}
        if isinstance(input_text, list) and "padding" not in call_kwargs:
            call_kwargs["padding"] = True
        call_kwargs.setdefault("return_tensors", "pt")
        tokenized = self.tokenizer(input_text, **call_kwargs)
        if self.device is not None:
            tokenized = tokenized.to(self.device)
        return GuardrailPreprocessOutput(data=tokenized)

    def infer(self, model_inputs: GuardrailPreprocessOutput[AnyDict]) -> GuardrailInferenceOutput[AnyDict]:
        """Run model inference on preprocessed inputs.

        For sequence-classification models (2D logits, shape ``(batch, num_classes)``)
        returns the uniform shape shared with other providers:

        - ``logits``: numpy array of raw model logits.
        - ``scores``: softmax (or sigmoid for multi-label) of logits.
        - ``predicted_indices``: list of argmax indices, one per row.
        - ``predicted_labels``: labels resolved via ``model.config.id2label``.
        - ``labels``: the full ordered label list from ``model.config.id2label``,
          aligning index-wise with each row of ``scores``.

        For models that emit higher-rank logits (e.g. causal LMs returning
        ``(batch, seq, vocab)``) the classification fields don't apply — there
        is no single "predicted class" per input. In that case ``logits`` is
        the raw torch tensor (so callers like ShieldGemma can index into it
        and run their own torch ops) and ``scores`` / ``predicted_indices`` /
        ``predicted_labels`` are ``None``.

        Args:
            model_inputs: The wrapped preprocessing output.

        Returns:
            GuardrailInferenceOutput wrapping the inference output. Always
            contains a ``logits`` key; ``scores`` / ``predicted_indices`` /
            ``predicted_labels`` may be ``None`` depending on the model shape.

        """
        with torch.no_grad():
            output = self.model(**model_inputs.data)
        raw_logits = output.logits.detach()
        if raw_logits.dim() != 2:
            # Causal-LM-style output, e.g. ShieldGemma where logits are
            # (batch, seq, vocab). Label resolution doesn't apply; return the
            # raw tensor so callers can slice it and run their own torch ops.
            return GuardrailInferenceOutput(
                data={
                    "logits": raw_logits,
                    "scores": None,
                    "predicted_indices": None,
                    "predicted_labels": None,
                    "labels": None,
                }
            )

        # ``.float()`` is a no-op for float32 but is required for bf16/fp16
        # logits, which numpy doesn't accept directly.
        logits = raw_logits.float().cpu().numpy()
        scores = _sigmoid(logits) if self.multi_label else _softmax(logits)
        predicted_indices = scores.argmax(axis=-1).tolist()
        id2label = self.model.config.id2label
        predicted_labels = [id2label[i] for i in predicted_indices]
        # Derive the class count from the logits, not from id2label: configs can
        # carry sparse or non-contiguous label maps, so fall back per index.
        labels = [id2label.get(i, f"LABEL_{i}") for i in range(logits.shape[-1])]
        return GuardrailInferenceOutput(
            data={
                "logits": logits,
                "scores": scores,
                "predicted_indices": predicted_indices,
                "predicted_labels": predicted_labels,
                "labels": labels,
            }
        )

    def generate_chat(
        self,
        messages: list[AnyDict],
        *,
        max_new_tokens: int,
        do_sample: bool = False,
        temperature: float | None = None,
        chat_template_kwargs: AnyDict | None = None,
        generation_kwargs: AnyDict | None = None,
        skip_special_tokens: bool = True,
        apply_chat_template: bool = True,
    ) -> GuardrailInferenceOutput[AnyDict]:
        """Apply the model's chat template and run ``model.generate`` under ``no_grad``.

        Defaults match what decoder-LLM guardrails (GraniteGuardian, LlamaGuard)
        previously wired inline: ``add_generation_prompt=True``, ``tokenize=True``,
        ``return_dict=True``, ``return_tensors="pt"``. Callers can override any
        of these via ``chat_template_kwargs`` (e.g. ``documents``,
        ``available_tools``, or ``add_generation_prompt=False`` for models that
        don't want an assistant prefix).

        Set ``apply_chat_template=False`` to feed ``messages[0]["content"]`` to the
        model verbatim (for models shipping their own instruction wrapper, e.g.
        WildGuard). Set ``skip_special_tokens=False`` to keep special tokens in the
        decoded output (for models whose verdict is a special token, e.g. Kanana).
        """
        if apply_chat_template:
            template_kwargs: AnyDict = {
                "add_generation_prompt": True,
                "tokenize": True,
                "return_dict": True,
                "return_tensors": "pt",
                **(chat_template_kwargs or {}),
            }
            inputs = self.tokenizer.apply_chat_template(messages, **template_kwargs)
        else:
            inputs = self.tokenizer(messages[0]["content"], return_tensors="pt")
        if self.device is not None and hasattr(inputs, "to"):
            inputs = inputs.to(self.device)

        prompt_len = int(inputs["input_ids"].shape[-1])

        gen_kwargs: AnyDict = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
        if do_sample and temperature is not None:
            gen_kwargs["temperature"] = temperature
        if generation_kwargs:
            gen_kwargs.update(generation_kwargs)

        with torch.no_grad():
            output = self.model.generate(**inputs, **gen_kwargs)

        generated = output[:, prompt_len:]
        text: str = self.tokenizer.decode(generated[0], skip_special_tokens=skip_special_tokens)

        return GuardrailInferenceOutput(
            data={
                "generated_text": text,
                "prompt_token_count": prompt_len,
                "completion_token_count": int(generated.shape[-1]),
                "raw": output,
            }
        )
