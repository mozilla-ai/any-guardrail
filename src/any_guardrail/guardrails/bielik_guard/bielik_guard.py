from collections.abc import Sequence
from typing import Any, ClassVar

from any_guardrail.base import StandardGuardrail
from any_guardrail.guardrails.utils import default
from any_guardrail.providers.base import StandardProvider
from any_guardrail.providers.huggingface import HuggingFaceProvider
from any_guardrail.types import CategoryResult, GuardrailOutput, StandardInferenceOutput, StandardPreprocessOutput

BIELIK_DEFAULT_THRESHOLD = 0.5

# The 0.5B variant ships custom modeling code (config ``auto_map``); the 0.1B variant
# is a plain RoBERTa classifier. Only the former needs ``trust_remote_code``.
_TRUST_REMOTE_CODE_MODELS = frozenset({"speakleash/Bielik-Guard-0.5B-v1.1"})


def _build_output(scores_row: Sequence[float], labels: Sequence[str] | None, threshold: float) -> GuardrailOutput:
    probabilities = [float(probability) for probability in scores_row]
    # Category names come from the model's id2label (multi-label heads expose them
    # via the provider's ``labels``); the published Bielik card does not fix an index order.
    names = list(labels) if labels is not None else [f"LABEL_{index}" for index in range(len(probabilities))]
    categories = [
        CategoryResult(name=name, score=probability, triggered=probability > threshold)
        for name, probability in zip(names, probabilities, strict=True)
    ]
    return GuardrailOutput(
        valid=not any(category.triggered for category in categories),
        score=max(probabilities) if probabilities else None,
        categories=categories,
    )


class BielikGuard(StandardGuardrail):
    """Bielik Guard — Polish multi-label safety classifier (SpeakLeash / Bielik.AI).

    Encoder classifier emitting independent per-category probabilities (sigmoid) over
    five safety categories: Hate/Aggression, Vulgarities, Sexual Content, Crime, and
    Self-Harm. ``valid`` is ``True`` when no category exceeds ``threshold``; ``score``
    is the maximum category probability; ``categories`` carries every category. The
    repos are gated (auto-approve) — authenticate with ``hf auth login`` first.

    For more information, please see the model cards:

    - [Bielik-Guard-0.1B-v1.1](https://huggingface.co/speakleash/Bielik-Guard-0.1B-v1.1) (default).
    - [Bielik-Guard-0.5B-v1.1](https://huggingface.co/speakleash/Bielik-Guard-0.5B-v1.1).

    Args:
        model_id: Optional HuggingFace model ID. Defaults to the 0.1B variant.
        threshold: Per-category probability above which a category is flagged. Defaults to 0.5.
        provider: Optional pre-configured provider. Defaults to a ``HuggingFaceProvider``
            with ``multi_label=True``.

    """

    SUPPORTED_MODELS: ClassVar = [
        "speakleash/Bielik-Guard-0.1B-v1.1",
        "speakleash/Bielik-Guard-0.5B-v1.1",
    ]

    def __init__(
        self,
        model_id: str | None = None,
        threshold: float = BIELIK_DEFAULT_THRESHOLD,
        provider: StandardProvider | None = None,
    ) -> None:
        """Initialize the Bielik Guard guardrail."""
        self.model_id = default(model_id, self.SUPPORTED_MODELS)
        self.threshold = threshold
        self.provider = provider or HuggingFaceProvider(
            multi_label=True,
            trust_remote_code=self.model_id in _TRUST_REMOTE_CODE_MODELS,
        )
        self.provider.load_model(self.model_id)

    def _pre_processing(self, input_text: str) -> StandardPreprocessOutput:
        return self.provider.pre_process(input_text, truncation=True)

    def _inference(self, model_inputs: StandardPreprocessOutput) -> StandardInferenceOutput:
        return self.provider.infer(model_inputs)

    def _post_processing(self, model_outputs: StandardInferenceOutput) -> GuardrailOutput:
        data = model_outputs.data
        return _build_output(data["scores"][0], data.get("labels"), self.threshold)

    def _validate_batch(self, input_texts: list[str], **kwargs: Any) -> list[GuardrailOutput]:
        model_inputs = self.provider.pre_process(input_texts, truncation=True, **kwargs)
        data = self.provider.infer(model_inputs).data
        labels = data.get("labels")
        return [_build_output(row, labels, self.threshold) for row in data["scores"]]
