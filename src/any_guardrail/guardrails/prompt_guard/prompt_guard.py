from collections.abc import Sequence
from typing import Any, ClassVar

from any_guardrail.base import StandardGuardrail
from any_guardrail.guardrails.utils import default
from any_guardrail.providers.base import StandardProvider
from any_guardrail.providers.huggingface import HuggingFaceProvider
from any_guardrail.types import CategoryResult, GuardrailOutput, StandardInferenceOutput, StandardPreprocessOutput

# Llama Prompt Guard 2 is a binary classifier: index 0 = benign, index 1 = malicious
# (jailbreak / prompt injection). Keyed by index rather than label string because the
# repos are gated and the published id2label is the generic ``LABEL_0``/``LABEL_1``.
MALICIOUS_INDEX = 1


def _build_output(scores_row: Sequence[float], predicted_index: int, labels: Sequence[str] | None) -> GuardrailOutput:
    probabilities = [float(probability) for probability in scores_row]
    names = list(labels) if labels is not None else [f"LABEL_{index}" for index in range(len(probabilities))]
    categories = [
        CategoryResult(name=name, score=probability, triggered=index == predicted_index)
        for index, (name, probability) in enumerate(zip(names, probabilities, strict=True))
    ]
    malicious_score = probabilities[MALICIOUS_INDEX] if len(probabilities) > MALICIOUS_INDEX else None
    return GuardrailOutput(valid=predicted_index != MALICIOUS_INDEX, score=malicious_score, categories=categories)


class PromptGuard(StandardGuardrail):
    """Meta Llama Prompt Guard 2 — encoder classifier for prompt injection / jailbreak detection.

    Binary DeBERTa classifier that labels a prompt ``benign`` or ``malicious``.
    ``valid`` is ``True`` when the prompt is benign; ``score`` is the malicious
    probability. The repos are gated under the Llama 4 Community License — accept
    the terms and authenticate with ``hf auth login`` before first use.

    For more information, please see the model cards:

    - [Llama-Prompt-Guard-2-86M](https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M)
      (default) — mDeBERTa-base, multilingual.
    - [Llama-Prompt-Guard-2-22M](https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-22M)
      — DeBERTa-xsmall, English, ~75% lower latency.
    """

    SUPPORTED_MODELS: ClassVar = [
        "meta-llama/Llama-Prompt-Guard-2-86M",
        "meta-llama/Llama-Prompt-Guard-2-22M",
    ]

    def __init__(self, model_id: str | None = None, provider: StandardProvider | None = None) -> None:
        """Initialize the Prompt Guard 2 guardrail."""
        self.model_id = default(model_id, self.SUPPORTED_MODELS)
        self.provider = provider or HuggingFaceProvider()
        self.provider.load_model(self.model_id)

    def _pre_processing(self, input_text: str) -> StandardPreprocessOutput:
        return self.provider.pre_process(input_text, truncation=True)

    def _inference(self, model_inputs: StandardPreprocessOutput) -> StandardInferenceOutput:
        return self.provider.infer(model_inputs)

    def _post_processing(self, model_outputs: StandardInferenceOutput) -> GuardrailOutput:
        data = model_outputs.data
        return _build_output(data["scores"][0], int(data["predicted_indices"][0]), data.get("labels"))

    def _validate_batch(self, input_texts: list[str], **kwargs: Any) -> list[GuardrailOutput]:
        model_inputs = self.provider.pre_process(input_texts, truncation=True, **kwargs)
        data = self.provider.infer(model_inputs).data
        labels = data.get("labels")
        return [
            _build_output(row, int(predicted_index), labels)
            for row, predicted_index in zip(data["scores"], data["predicted_indices"], strict=True)
        ]
