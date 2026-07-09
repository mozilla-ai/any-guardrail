from typing import Any, ClassVar

from any_guardrail.base import StandardGuardrail
from any_guardrail.guardrails.utils import default, match_injection_label, match_injection_label_batch
from any_guardrail.providers.base import StandardProvider
from any_guardrail.providers.huggingface import HuggingFaceProvider
from any_guardrail.types import GuardrailOutput, StandardInferenceOutput, StandardPreprocessOutput

PANGOLIN_INJECTION_LABEL = "unsafe"


class Pangolin(StandardGuardrail):
    """Pangolin Guard — binary prompt-injection classifier built on ModernBERT (dcarpintero).

    Runs one of dcarpintero's ModernBERT encoder classifiers over a single user prompt and reports
    whether the text is a prompt-injection / jailbreak attempt. Each model is a two-class sequence
    classifier whose unsafe class is labeled ``"unsafe"``; the guardrail treats that class as the
    risky one. The default ``pangolin-guard-base`` is ModernBERT-base; ``pangolin-guard-large`` is a
    higher-accuracy ModernBERT-large drop-in with an 8192-token context and the same ``"unsafe"``
    label.

    Expected input: prompt-only text. ``validate(input_text)`` accepts a single string, or a
    ``list[str]`` to classify a batch in one call; there is no prompt+response or chat-message mode.

    Verdict mapping onto ``GuardrailOutput``:

    - ``valid`` is ``True`` when the predicted class is not ``"unsafe"`` (i.e. the text looks safe).
    - ``score`` is the model's probability of the ``"unsafe"`` class (canonical risk direction:
      higher = riskier).
    - ``categories`` carries one ``CategoryResult`` per class label, each with its softmax ``score``
      and a ``triggered`` flag marking the argmax class.
    - No ``spans`` or ``modified_text`` are produced.

    For more information, see:

    - [Pangolin Guard base](https://huggingface.co/dcarpintero/pangolin-guard-base) (default) — ModernBERT-base.
    - [Pangolin Guard large](https://huggingface.co/dcarpintero/pangolin-guard-large) — ModernBERT-large;
      higher accuracy, 8192-token context. Same ``"unsafe"`` label, so it is a drop-in alternative.

    """

    SUPPORTED_MODELS: ClassVar = ["dcarpintero/pangolin-guard-base", "dcarpintero/pangolin-guard-large"]

    def __init__(self, model_id: str | None = None, provider: StandardProvider | None = None) -> None:
        """Initialize the Pangolin Guard guardrail.

        Args:
            model_id: Optional HuggingFace model ID. Must be one of ``SUPPORTED_MODELS``; defaults
                to ``dcarpintero/pangolin-guard-base`` (ModernBERT-base). Pass
                ``"dcarpintero/pangolin-guard-large"`` for the higher-accuracy ModernBERT-large
                variant with an 8192-token context. Both share the same ``"unsafe"`` label.
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
        return match_injection_label(model_outputs, PANGOLIN_INJECTION_LABEL)

    def _validate_batch(self, input_texts: list[str], **kwargs: Any) -> list[GuardrailOutput]:
        model_inputs = self.provider.pre_process(input_texts, **kwargs)
        model_outputs = self.provider.infer(model_inputs)
        return match_injection_label_batch(model_outputs, PANGOLIN_INJECTION_LABEL)
