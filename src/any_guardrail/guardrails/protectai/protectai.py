from typing import Any, ClassVar

from any_guardrail.base import GuardrailName, StandardGuardrail
from any_guardrail.guardrails.utils import default, match_injection_label, match_injection_label_batch
from any_guardrail.providers.base import StandardProvider
from any_guardrail.providers.huggingface import HuggingFaceProvider
from any_guardrail.registry import GUARDRAIL_METADATA
from any_guardrail.taxonomy import GuardrailMetadata
from any_guardrail.types import GuardrailOutput, StandardInferenceOutput, StandardPreprocessOutput

PROTECTAI_INJECTION_LABEL = "INJECTION"


class Protectai(StandardGuardrail):
    """ProtectAI — binary prompt-injection classifiers built on DeBERTa-v3 and DistilRoBERTa (ProtectAI).

    Runs one of ProtectAI's encoder classifiers over a single user prompt and reports whether the
    text is a prompt-injection / jailbreak attempt. Each model is a two-class sequence classifier
    whose unsafe class is labeled ``"INJECTION"``; the guardrail treats that class as the risky one.
    ``SUPPORTED_MODELS`` spans ProtectAI's LLM-security collection — three DeBERTa-v3 prompt-injection
    models (small, base v1, base v2) plus a DistilRoBERTa "rejection" detector — with
    ``deberta-v3-small-prompt-injection-v2`` as the default.

    Expected input: prompt-only text. ``validate(input_text)`` accepts a single string, or a
    ``list[str]`` to classify a batch in one call; there is no prompt+response or chat-message mode.

    Verdict mapping onto ``GuardrailOutput``:

    - ``valid`` is ``True`` when the predicted class is not ``"INJECTION"`` (i.e. the text looks safe).
    - ``score`` is the model's probability of the ``"INJECTION"`` class (canonical risk direction:
      higher = riskier).
    - ``categories`` carries one ``CategoryResult`` per class label, each with its softmax ``score``
      and a ``triggered`` flag marking the argmax class.
    - No ``spans`` or ``modified_text`` are produced.

    For more information, see:

    - [ProtectAI LLM-security collection](https://huggingface.co/collections/protectai/llm-security-65c1f17a11c4251eeab53f40)
    - [deberta-v3-small-prompt-injection-v2](https://huggingface.co/ProtectAI/deberta-v3-small-prompt-injection-v2) (default)
    - [distilroberta-base-rejection-v1](https://huggingface.co/ProtectAI/distilroberta-base-rejection-v1)
    - [deberta-v3-base-prompt-injection](https://huggingface.co/ProtectAI/deberta-v3-base-prompt-injection)
    - [deberta-v3-base-prompt-injection-v2](https://huggingface.co/ProtectAI/deberta-v3-base-prompt-injection-v2)

    """

    SUPPORTED_MODELS: ClassVar = [
        "ProtectAI/deberta-v3-small-prompt-injection-v2",
        "ProtectAI/distilroberta-base-rejection-v1",
        "ProtectAI/deberta-v3-base-prompt-injection",
        "ProtectAI/deberta-v3-base-prompt-injection-v2",
    ]

    METADATA: ClassVar[GuardrailMetadata] = GUARDRAIL_METADATA[GuardrailName.PROTECTAI]

    def __init__(self, model_id: str | None = None, provider: StandardProvider | None = None) -> None:
        """Initialize the ProtectAI guardrail.

        Args:
            model_id: Optional HuggingFace model ID. Must be one of ``SUPPORTED_MODELS``; defaults
                to ``ProtectAI/deberta-v3-small-prompt-injection-v2``. Pass one of the other entries
                (e.g. ``"ProtectAI/deberta-v3-base-prompt-injection-v2"``) for a larger DeBERTa-v3
                classifier, or ``"ProtectAI/distilroberta-base-rejection-v1"`` for the rejection head.
            provider: Execution backend that loads the model and runs inference. Defaults to a
                ``HuggingFaceProvider`` targeting ``AutoModelForSequenceClassification``. Supply your
                own to control device, dtype, or ``cache_dir``, or to run against a different backend.

        Raises:
            ValueError: If ``model_id`` is not in ``SUPPORTED_MODELS``.

        """
        self.model_id = default(model_id, self.SUPPORTED_MODELS)
        self.provider = provider or HuggingFaceProvider()
        self.provider.load_model(self.model_id)

    def _pre_processing(self, input_text: str) -> StandardPreprocessOutput:
        return self.provider.pre_process(input_text)

    def _inference(self, model_inputs: StandardPreprocessOutput) -> StandardInferenceOutput:
        return self.provider.infer(model_inputs)

    def _post_processing(self, model_outputs: StandardInferenceOutput) -> GuardrailOutput:
        return match_injection_label(model_outputs, PROTECTAI_INJECTION_LABEL)

    def _validate_batch(self, input_texts: list[str], **kwargs: Any) -> list[GuardrailOutput]:
        model_inputs = self.provider.pre_process(input_texts, **kwargs)
        model_outputs = self.provider.infer(model_inputs)
        return match_injection_label_batch(model_outputs, PROTECTAI_INJECTION_LABEL)
