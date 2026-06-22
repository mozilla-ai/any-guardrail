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
        """Generate a chat completion from a list of messages.

        Opt-in capability for chat-style decoder LLM providers. The default
        raises :class:`NotImplementedError` so encoder-only or API-shaped
        providers (Encoderfile, AzureContentSafety, etc.) are unaffected.

        Implementations should return a uniform shape so guardrails consuming
        this method are provider-agnostic:

        - ``generated_text``: decoded new tokens only (prompt stripped).
        - ``prompt_token_count``: prompt length in tokens, or ``None`` when the
          provider tokenizes server-side and doesn't surface it.
        - ``completion_token_count``: generated length in tokens, or ``None``.
        - ``raw``: provider-specific raw output (HF tensor, OpenAI JSON, etc.).

        Args:
            messages: Chat messages as a list of ``{"role": ..., "content": ...}``
                dicts. ``content`` may be a string or a list of multimodal parts
                depending on the model.
            max_new_tokens: Maximum number of tokens to generate.
            do_sample: When ``False`` (default), generate greedily.
            temperature: Sampling temperature. Ignored when ``do_sample`` is
                ``False``.
            chat_template_kwargs: Extra kwargs forwarded to the model's chat
                template (e.g. ``documents``, ``available_tools``).
            generation_kwargs: Extra kwargs forwarded to the underlying
                generation call (e.g. ``pad_token_id`` for HF, OpenAI-specific
                fields for HTTP backends).
            skip_special_tokens: Whether to strip special tokens when decoding.
                Set ``False`` for guardrails whose verdict *is* a special token
                (e.g. Kanana Safeguard's ``<SAFE>``/``<UNSAFE-*>``).
            apply_chat_template: When ``True`` (default), render ``messages``
                through the model's chat template. Set ``False`` to feed the
                first message's ``content`` to the model as a raw prompt, for
                models shipping their own instruction wrapper (e.g. WildGuard).

        """
        msg = (
            f"{type(self).__name__} does not support generate_chat(); "
            f"use a chat-capable provider such as HuggingFaceProvider or LlamafileProvider."
        )
        raise NotImplementedError(msg)


# Standard provider type alias
StandardProvider = Provider[AnyDict, AnyDict]
"""Type alias for providers using standard AnyDict inputs/outputs."""
