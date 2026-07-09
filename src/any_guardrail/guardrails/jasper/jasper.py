from typing import Any, ClassVar

from any_guardrail.base import StandardGuardrail
from any_guardrail.guardrails.utils import default, match_injection_label, match_injection_label_batch
from any_guardrail.providers.base import StandardProvider
from any_guardrail.providers.huggingface import HuggingFaceProvider
from any_guardrail.types import GuardrailOutput, StandardInferenceOutput, StandardPreprocessOutput

JASPER_INJECTION_LABEL = "INJECTION"


class Jasper(StandardGuardrail):
    """Jasper — binary prompt-injection classifiers built on DeBERTa-v3-base and gELECTRA-base (JasperLS).

    Runs one of JasperLS's encoder classifiers over a single user prompt and reports whether the
    text is a prompt-injection attempt. Each model is a two-class sequence classifier whose unsafe
    class is labeled ``"INJECTION"``; the guardrail treats that class as the risky one. The default
    ``gelectra-base-injection`` is built on a German-language gELECTRA encoder, while
    ``deberta-v3-base-injection`` is built on an English DeBERTa-v3 encoder — pick the one that
    matches your prompt language.

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

    - [gelectra-base-injection](https://huggingface.co/JasperLS/gelectra-base-injection) (default)
    - [deberta-v3-base-injection](https://huggingface.co/JasperLS/deberta-v3-base-injection)

    Args:
        model_id: Optional HuggingFace model ID. Must be one of ``SUPPORTED_MODELS``; defaults to
            ``JasperLS/gelectra-base-injection``.
        provider: Execution backend that loads the model and runs inference. Defaults to a
            ``HuggingFaceProvider`` targeting ``AutoModelForSequenceClassification``.

    Raises:
        ValueError: If ``model_id`` is not in ``SUPPORTED_MODELS``.

    """

    SUPPORTED_MODELS: ClassVar = ["JasperLS/gelectra-base-injection", "JasperLS/deberta-v3-base-injection"]

    def __init__(self, model_id: str | None = None, provider: StandardProvider | None = None) -> None:
        """Initialize the Jasper guardrail.

        Args:
            model_id: Optional HuggingFace model ID. Must be one of ``SUPPORTED_MODELS``; defaults
                to ``JasperLS/gelectra-base-injection`` (a German-language gELECTRA encoder). Pass
                ``"JasperLS/deberta-v3-base-injection"`` for the English DeBERTa-v3 variant.
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
        return match_injection_label(model_outputs, JASPER_INJECTION_LABEL)

    def _validate_batch(self, input_texts: list[str], **kwargs: Any) -> list[GuardrailOutput]:
        model_inputs = self.provider.pre_process(input_texts, **kwargs)
        model_outputs = self.provider.infer(model_inputs)
        return match_injection_label_batch(model_outputs, JASPER_INJECTION_LABEL)
