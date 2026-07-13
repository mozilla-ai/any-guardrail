from typing import Any, ClassVar

from any_guardrail.base import GuardrailName, StandardGuardrail
from any_guardrail.guardrails.utils import default, match_injection_label, match_injection_label_batch
from any_guardrail.providers.base import StandardProvider
from any_guardrail.providers.huggingface import HuggingFaceProvider
from any_guardrail.registry import GUARDRAIL_METADATA
from any_guardrail.taxonomy import GuardrailMetadata
from any_guardrail.types import GuardrailOutput, StandardInferenceOutput, StandardPreprocessOutput

DEEPSET_INJECTION_LABEL = "INJECTION"


class Deepset(StandardGuardrail):
    """Binary prompt-injection classifier.

    Encoder sequence classifier that labels a text as ``LEGIT`` or ``INJECTION``.
    Feed it the raw user prompt (or any untrusted text such as retrieved snippets);
    no model response or extra context is involved. ``validate`` accepts a single
    string or a ``list[str]``, in which case the whole list runs as one true
    batched inference call and a list of outputs is returned in the same order.

    Verdict mapping onto ``GuardrailOutput``:

    - ``valid`` is ``True`` when the predicted label is not ``INJECTION``.
    - ``score`` is the probability of the ``INJECTION`` class (canonical risk
      direction: higher = riskier), regardless of which label won the argmax.
    - ``categories`` holds the full label distribution — one entry per class with
      its probability; the predicted class is marked ``triggered=True``.

    For more information, see:

    - [deepset/deberta-v3-base-injection model card](https://huggingface.co/deepset/deberta-v3-base-injection).
    """

    SUPPORTED_MODELS: ClassVar = ["deepset/deberta-v3-base-injection"]

    METADATA: ClassVar[GuardrailMetadata] = GUARDRAIL_METADATA[GuardrailName.DEEPSET]

    def __init__(self, model_id: str | None = None, provider: StandardProvider | None = None) -> None:
        """Initialize the Deepset guardrail."""
        self.model_id = default(model_id, self.SUPPORTED_MODELS)
        self.provider = provider or HuggingFaceProvider()
        self.provider.load_model(self.model_id)

    def _pre_processing(self, input_text: str) -> StandardPreprocessOutput:
        return self.provider.pre_process(input_text)

    def _inference(self, model_inputs: StandardPreprocessOutput) -> StandardInferenceOutput:
        return self.provider.infer(model_inputs)

    def _post_processing(self, model_outputs: StandardInferenceOutput) -> GuardrailOutput:
        return match_injection_label(model_outputs, DEEPSET_INJECTION_LABEL)

    def _validate_batch(self, input_texts: list[str], **kwargs: Any) -> list[GuardrailOutput]:
        model_inputs = self.provider.pre_process(input_texts, **kwargs)
        model_outputs = self.provider.infer(model_inputs)
        return match_injection_label_batch(model_outputs, DEEPSET_INJECTION_LABEL)
