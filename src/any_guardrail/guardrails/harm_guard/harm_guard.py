from typing import ClassVar

from any_guardrail.base import GuardrailOutput, StandardGuardrail
from any_guardrail.guardrails.utils import default
from any_guardrail.providers.base import StandardProvider
from any_guardrail.providers.huggingface import HuggingFaceProvider
from any_guardrail.types import (
    CategoryResult,
    GuardrailPreprocessOutput,
    StandardInferenceOutput,
    StandardPreprocessOutput,
)

HARMGUARD_DEFAULT_THRESHOLD = 0.5  # Taken from the HarmGuard paper


class HarmGuard(StandardGuardrail):
    """HarmAug-Guard — binary safety and jailbreak classifier built on DeBERTa-v3-large, scoring a prompt or prompt-response pair.

    HarmAug-Guard is a 435M DeBERTa-v3-large classifier distilled from a much larger (7B+)
    teacher safety model using the HarmAug data-augmentation method, which jailbreaks an LLM
    to synthesize harmful instructions for training. It classifies whether an LLM interaction
    is safe or unsafe and flags jailbreak attempts, reaching an F1 comparable to 7B+ safety
    models at a fraction of the compute.

    Two input shapes are supported through ``validate``:

    - a single prompt string — judges the prompt on its own; or
    - a prompt plus a response (``output_text``) — tokenized as a text pair so the response is
      judged in the context of the prompt.

    Verdict mapping onto ``GuardrailOutput``:

    - ``score`` (canonical risk: higher = riskier) is the ``unsafe`` probability
      (``0.0`` = safe, ``1.0`` = unsafe), taken from ``softmax(logits)[1]`` — or from the
      column the provider labels ``unsafe`` when label names are available.
    - ``valid`` is ``True`` when that unsafe probability is below ``threshold`` (default 0.5,
      from the paper).
    - ``categories`` carries a ``safe`` and an ``unsafe`` entry with their probabilities and
      ``triggered`` flags.

    For more information, see:

    - [HarmAug-Guard model card](https://huggingface.co/hbseong/HarmAug-Guard).
    - [HarmAug: Effective Data Augmentation for Knowledge Distillation of Safety Guard Models (arXiv:2410.01524)](https://arxiv.org/abs/2410.01524).

    Args:
        model_id: Optional HuggingFace model ID from ``SUPPORTED_MODELS``. Defaults to
            ``hbseong/HarmAug-Guard``.
        threshold: Unsafe-probability cutoff at or above which the input is flagged unsafe
            (``valid=False``). Defaults to 0.5, the value used in the HarmAug paper.
        provider: Optional pre-configured provider. Defaults to a ``HuggingFaceProvider``.

    """

    SUPPORTED_MODELS: ClassVar = ["hbseong/HarmAug-Guard"]

    def __init__(
        self,
        model_id: str | None = None,
        threshold: float = HARMGUARD_DEFAULT_THRESHOLD,
        provider: StandardProvider | None = None,
    ) -> None:
        """Initialize the HarmGuard guardrail.

        Args:
            model_id: Optional HuggingFace model ID. Must be one of ``SUPPORTED_MODELS``;
                defaults to ``hbseong/HarmAug-Guard``.
            threshold: Unsafe-probability cutoff at or above which the input is flagged unsafe
                (``valid=False``). Defaults to 0.5, the value used in the HarmAug paper; lower
                it to catch borderline content, raise it to reduce false positives.
            provider: Optional pre-configured provider. If ``None``, a default
                ``HuggingFaceProvider`` is built and the model is loaded eagerly.

        Raises:
            ValueError: If ``model_id`` is not in ``SUPPORTED_MODELS``.

        """
        self.model_id = default(model_id, self.SUPPORTED_MODELS)
        self.threshold = threshold
        self.provider = provider or HuggingFaceProvider()
        self.provider.load_model(self.model_id)

    def validate(self, input_text: str, output_text: str | None = None) -> GuardrailOutput:  # type: ignore[override]
        """Validate whether the input (and optionally the response) is safe.

        Args:
            input_text: The prompt / user text to evaluate, e.g. ``"How do I pick a lock?"``.
                A single string; list/batch input is not supported by this override.
            output_text: Optional model response. When provided, ``input_text`` and
                ``output_text`` are tokenized as a text pair so the response is judged for
                safety in the context of the prompt; when ``None``, only the prompt is judged.

        Returns:
            GuardrailOutput with ``valid=True`` when safe and ``valid=False`` when the unsafe
            probability reaches ``threshold``. ``score`` is that unsafe probability (higher =
            riskier; ``0.0`` = safe, ``1.0`` = unsafe), and ``categories`` holds the ``safe`` /
            ``unsafe`` probabilities.

        """
        return self._execute(input_text, output_text)

    def _pre_processing(self, input_text: str, output_text: str | None = None) -> StandardPreprocessOutput:
        """Tokenize the prompt and, optionally, the response as a text pair.

        Args:
            input_text: The prompt / input text to tokenize.
            output_text: Optional response text. When provided, the tokenizer builds a
                text-pair input (``input_text``, ``output_text``) so the response is evaluated
                in the context of the prompt; when ``None``, only ``input_text`` is tokenized.

        Returns:
            GuardrailPreprocessOutput wrapping the tokenized tensors, moved to the provider's
            device when it exposes one.

        """
        if output_text is None:
            tokenized = self.provider.tokenizer(input_text, return_tensors="pt")  # type: ignore[attr-defined]
        else:
            tokenized = self.provider.tokenizer(input_text, output_text, return_tensors="pt")  # type: ignore[attr-defined]
        device = getattr(self.provider, "device", None)
        if device is not None:
            tokenized = tokenized.to(device)
        return GuardrailPreprocessOutput(data=tokenized)

    def _inference(self, model_inputs: StandardPreprocessOutput) -> StandardInferenceOutput:
        return self.provider.infer(model_inputs)

    # Index of the "unsafe" class when the model exposes no usable label names.
    # HarmAug-Guard ships no id2label, and its card defines softmax(logits)[1]
    # as the unsafe probability.
    _UNSAFE_INDEX = 1

    def _post_processing(self, model_outputs: StandardInferenceOutput) -> GuardrailOutput:
        scores_row = [float(probability) for probability in model_outputs.data["scores"][0]]
        # Prefer resolving the unsafe class by the provider's label list (the
        # reason infer() surfaces ``labels``); fall back to the documented column
        # for models like HarmAug-Guard that don't ship meaningful label names.
        labels = model_outputs.data.get("labels")
        unsafe_index = self._UNSAFE_INDEX
        if labels is not None:
            unsafe_index = next(
                (i for i, name in enumerate(labels) if name.lower() == "unsafe"),
                self._UNSAFE_INDEX,
            )
        safe_index = 1 - unsafe_index  # HarmGuard is a binary safe/unsafe classifier.

        final_score = scores_row[unsafe_index]
        triggered = final_score >= self.threshold
        return GuardrailOutput(
            valid=not triggered,
            score=final_score,
            categories=[
                CategoryResult(name="safe", score=scores_row[safe_index], triggered=not triggered),
                CategoryResult(name="unsafe", score=final_score, triggered=triggered),
            ],
        )
