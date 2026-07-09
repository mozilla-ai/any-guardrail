from typing import Any, ClassVar

from any_guardrail.base import StandardGuardrail
from any_guardrail.guardrails.utils import default, match_injection_label, match_injection_label_batch
from any_guardrail.providers.base import StandardProvider
from any_guardrail.providers.huggingface import HuggingFaceProvider
from any_guardrail.types import GuardrailOutput, StandardInferenceOutput, StandardPreprocessOutput

SENTINEL_INJECTION_LABEL = "jailbreak"


class Sentinel(StandardGuardrail):
    """Sentinel — binary prompt-injection classifier built on DeBERTa (Qualifire).

    Runs Qualifire's DeBERTa-based encoder classifier over a single user prompt and reports whether
    the text is a prompt-injection / jailbreak attempt. The model is a two-class sequence classifier
    whose unsafe class is labeled ``"jailbreak"``; the guardrail treats that class as the risky one.

    Expected input: prompt-only text. ``validate(input_text)`` accepts a single string, or a
    ``list[str]`` to classify a batch in one call; there is no prompt+response or chat-message mode.

    Verdict mapping onto ``GuardrailOutput``:

    - ``valid`` is ``True`` when the predicted class is not ``"jailbreak"`` (i.e. the text looks safe).
    - ``score`` is the model's probability of the ``"jailbreak"`` class (canonical risk direction:
      higher = riskier).
    - ``categories`` carries one ``CategoryResult`` per class label, each with its softmax ``score``
      and a ``triggered`` flag marking the argmax class.
    - No ``spans`` or ``modified_text`` are produced.

    For more information, see:

    - [Sentinel model card](https://huggingface.co/qualifire/prompt-injection-sentinel)

    """

    SUPPORTED_MODELS: ClassVar = ["qualifire/prompt-injection-sentinel"]

    def __init__(self, model_id: str | None = None, provider: StandardProvider | None = None) -> None:
        """Initialize the Sentinel guardrail.

        Args:
            model_id: Optional HuggingFace model ID. Must be one of ``SUPPORTED_MODELS``; defaults
                to (and currently only supports) ``qualifire/prompt-injection-sentinel``.
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
        return match_injection_label(model_outputs, SENTINEL_INJECTION_LABEL)

    def _validate_batch(self, input_texts: list[str], **kwargs: Any) -> list[GuardrailOutput]:
        model_inputs = self.provider.pre_process(input_texts, **kwargs)
        model_outputs = self.provider.infer(model_inputs)
        return match_injection_label_batch(model_outputs, SENTINEL_INJECTION_LABEL)
