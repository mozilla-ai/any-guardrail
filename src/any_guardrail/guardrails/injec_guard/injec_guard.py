from typing import Any, ClassVar

from any_guardrail.base import StandardGuardrail
from any_guardrail.guardrails.utils import default, match_injection_label, match_injection_label_batch
from any_guardrail.providers.base import StandardProvider
from any_guardrail.providers.huggingface import HuggingFaceProvider
from any_guardrail.types import GuardrailOutput, StandardInferenceOutput, StandardPreprocessOutput

INJECGUARD_LABEL = "injection"


class InjecGuard(StandardGuardrail):
    """PIGuard — binary prompt-injection classifier built on DeBERTa-v3 and trained to mitigate over-defense (successor to InjecGuard).

    Runs PIGuard's DeBERTa-v3 encoder classifier over a single user prompt and reports whether the
    text is a prompt-injection attempt. The model is a two-class sequence classifier whose unsafe
    class is labeled ``"injection"``; the guardrail treats that class as the risky one. PIGuard adds
    the "Mitigating Over-defense for Free" (MOF) training strategy (ACL 2025), which reduces the
    trigger-word bias that makes prompt-injection guards falsely flag benign inputs.

    Expected input: prompt-only text. ``validate(input_text)`` accepts a single string, or a
    ``list[str]`` to classify a batch in one call; there is no prompt+response or chat-message mode.

    Verdict mapping onto ``GuardrailOutput``:

    - ``valid`` is ``True`` when the predicted class is not ``"injection"`` (i.e. the text looks safe).
    - ``score`` is the model's probability of the ``"injection"`` class (canonical risk direction:
      higher = riskier).
    - ``categories`` carries one ``CategoryResult`` per class label, each with its softmax ``score``
      and a ``triggered`` flag marking the argmax class.
    - No ``spans`` or ``modified_text`` are produced.

    ``leolee99/PIGuard`` is the renamed, maintained successor to InjecGuard (the rename was for
    licensing reasons); ``leolee99/InjecGuard`` is the original repository, kept for backward
    compatibility. Both share the same DeBERTa-v3 architecture and ``"injection"`` label and ship
    custom model code, so the default provider loads them with ``trust_remote_code=True``.

    For more information, see:

    - [PIGuard model card](https://huggingface.co/leolee99/PIGuard) (default)
    - [InjecGuard model card](https://huggingface.co/leolee99/InjecGuard)
    - [InjecGuard: Benchmarking and Mitigating Over-defense in Prompt Injection Guardrail Models (arXiv:2410.22770)](https://arxiv.org/abs/2410.22770)

    """

    SUPPORTED_MODELS: ClassVar = ["leolee99/PIGuard", "leolee99/InjecGuard"]

    def __init__(self, model_id: str | None = None, provider: StandardProvider | None = None) -> None:
        """Initialize the PIGuard guardrail.

        Args:
            model_id: Optional HuggingFace model ID. Must be one of ``SUPPORTED_MODELS``; defaults
                to ``leolee99/PIGuard`` (the maintained successor). Pass ``"leolee99/InjecGuard"``
                for the original repository, kept for backward compatibility.
            provider: Execution backend that loads the model and runs inference. Defaults to a
                ``HuggingFaceProvider`` constructed with ``trust_remote_code=True`` (PIGuard ships a
                custom model class), targeting ``AutoModelForSequenceClassification``. Supply your
                own to control device, dtype, or ``cache_dir``, or to run against a different backend.

        Raises:
            ValueError: If ``model_id`` is not in ``SUPPORTED_MODELS``.

        """
        self.model_id = default(model_id, self.SUPPORTED_MODELS)
        self.provider = provider or HuggingFaceProvider(trust_remote_code=True)
        self.provider.load_model(self.model_id)

    def _pre_processing(self, input_text: str) -> StandardPreprocessOutput:
        return self.provider.pre_process(input_text)

    def _inference(self, model_inputs: StandardPreprocessOutput) -> StandardInferenceOutput:
        return self.provider.infer(model_inputs)

    def _post_processing(self, model_outputs: StandardInferenceOutput) -> GuardrailOutput:
        return match_injection_label(model_outputs, INJECGUARD_LABEL)

    def _validate_batch(self, input_texts: list[str], **kwargs: Any) -> list[GuardrailOutput]:
        model_inputs = self.provider.pre_process(input_texts, **kwargs)
        model_outputs = self.provider.infer(model_inputs)
        return match_injection_label_batch(model_outputs, INJECGUARD_LABEL)
