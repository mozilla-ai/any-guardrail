from collections.abc import Sequence
from typing import Any, ClassVar

from any_guardrail.base import GuardrailName, StandardGuardrail
from any_guardrail.guardrails.utils import default
from any_guardrail.providers.base import StandardProvider
from any_guardrail.providers.huggingface import HuggingFaceProvider
from any_guardrail.registry import GUARDRAIL_METADATA
from any_guardrail.taxonomy import GuardrailMetadata
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
    """Encoder classifier for prompt-injection and jailbreak detection.

    Binary encoder classifier (mDeBERTa / DeBERTa) that labels a single prompt string
    ``benign`` (index 0) or ``malicious`` (index 1, i.e. a prompt-injection or jailbreak
    attempt). Meta's v2 collapsed Prompt Guard 1's separate "injection" class and focuses on
    explicit jailbreak / injection attacks. It screens prompt text only — it is not designed
    to judge model responses.

    Verdict mapping onto ``GuardrailOutput``:

    - ``valid`` is ``True`` when the predicted class is ``benign`` (argmax index != 1).
    - ``score`` (canonical risk: higher = riskier) is the ``malicious`` probability.
    - ``categories`` carries one ``CategoryResult`` per class, each with its probability and a
      ``triggered`` flag on the predicted class. The gated repos publish only the generic
      ``LABEL_0`` / ``LABEL_1`` names, so categories fall back to those when the provider
      surfaces no label list.

    Expected inputs: a single prompt string, or a ``list[str]`` for batched classification
    (the inherited ``validate`` dispatches list input to ``_validate_batch``). The 86M default
    is multilingual; the 22M variant is English-only.

    The repos are gated under the Llama 4 Community License — accept the terms on the model
    page and authenticate with ``hf auth login`` before first use.

    For more information, please see the model cards:

    - [Llama-Prompt-Guard-2-86M](https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M)
      (default) — mDeBERTa-base, multilingual.
    - [Llama-Prompt-Guard-2-22M](https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-22M)
      — DeBERTa-xsmall, English, ~75% lower latency.

    Args:
        model_id: Optional HuggingFace model ID from ``SUPPORTED_MODELS``. Defaults to
            ``meta-llama/Llama-Prompt-Guard-2-86M`` (multilingual); pass
            ``meta-llama/Llama-Prompt-Guard-2-22M`` for the smaller English-only variant.
        provider: Optional pre-configured provider. Defaults to a ``HuggingFaceProvider``
            (sequence-classification loader).

    """

    SUPPORTED_MODELS: ClassVar = [
        "meta-llama/Llama-Prompt-Guard-2-86M",
        "meta-llama/Llama-Prompt-Guard-2-22M",
    ]

    METADATA: ClassVar[GuardrailMetadata] = GUARDRAIL_METADATA[GuardrailName.PROMPT_GUARD]

    def __init__(self, model_id: str | None = None, provider: StandardProvider | None = None) -> None:
        """Initialize the Prompt Guard 2 guardrail.

        Args:
            model_id: Optional HuggingFace model ID. Must be one of ``SUPPORTED_MODELS``;
                defaults to ``meta-llama/Llama-Prompt-Guard-2-86M`` (mDeBERTa-base,
                multilingual). Pass ``meta-llama/Llama-Prompt-Guard-2-22M`` for the smaller
                English-only variant with ~75% lower latency.
            provider: Optional pre-configured provider. If ``None``, a default
                ``HuggingFaceProvider`` (targeting ``AutoModelForSequenceClassification``) is
                built and the model is loaded eagerly.

        Raises:
            ValueError: If ``model_id`` is not in ``SUPPORTED_MODELS``.

        """
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
