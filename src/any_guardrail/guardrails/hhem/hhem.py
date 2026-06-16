import time
from typing import Any, ClassVar

from any_guardrail.base import Guardrail
from any_guardrail.guardrails.utils import default
from any_guardrail.types import CategoryResult, GuardrailOutput, GuardrailUsage

HHEM_DEFAULT_THRESHOLD = 0.5


class Hhem(Guardrail):
    """Vectara HHEM-2.1-Open — hallucination / factual-consistency detector.

    A cross-encoder that scores how well a hypothesis (e.g. a generated answer) is
    supported by a premise (e.g. the source document), returning a consistency score in
    ``[0, 1]`` where higher means better supported. ``validate(input_text, context=...)``
    treats ``context`` as the premise and ``input_text`` as the hypothesis. ``valid`` is
    ``True`` when consistency ``>= threshold``; ``score`` is the canonical risk
    (``1 - consistency``, higher = more likely hallucinated).

    For more information, see the
    [HHEM-2.1-Open model card](https://huggingface.co/vectara/hallucination_evaluation_model).

    Args:
        threshold: Consistency at or above which the hypothesis is considered grounded.
        model_id: Optional HuggingFace model ID. Defaults to
            ``vectara/hallucination_evaluation_model``.
    """

    SUPPORTED_MODELS: ClassVar = ["vectara/hallucination_evaluation_model"]

    def __init__(self, threshold: float = HHEM_DEFAULT_THRESHOLD, model_id: str | None = None) -> None:
        """Initialize the HHEM guardrail."""
        self.model_id = default(model_id, self.SUPPORTED_MODELS)
        self.threshold = threshold
        from transformers import AutoModelForSequenceClassification

        # HHEM ships a custom model class exposing ``.predict([(premise, hypothesis), ...])``;
        # it does not fit the standard pre_process/infer provider shape, so load it directly.
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id, trust_remote_code=True)

    def validate(self, input_text: str, context: str | None = None, **kwargs: Any) -> GuardrailOutput:
        """Score how well ``input_text`` (hypothesis) is supported by ``context`` (premise).

        Args:
            input_text: The hypothesis/claim/answer to check.
            context: The premise/source the hypothesis must be grounded in. Required.
            **kwargs: Ignored.

        Returns:
            A :class:`GuardrailOutput` where ``valid`` is ``True`` when the consistency
            score is at or above ``threshold``, and ``score`` is ``1 - consistency``.

        """
        del kwargs
        if context is None:
            msg = "Hhem.validate requires `context` (the premise/source text) to score consistency against."
            raise ValueError(msg)
        start = time.perf_counter()
        consistency = float(self.model.predict([(context, input_text)])[0])
        result = GuardrailOutput(
            valid=consistency >= self.threshold,
            score=1.0 - consistency,
            categories=[CategoryResult(name="hallucination", score=1.0 - consistency, triggered=consistency < self.threshold)],
            extra={"consistency_score": consistency},
        )
        result.usage = GuardrailUsage(model_id=self.model_id, latency_ms=(time.perf_counter() - start) * 1000.0)
        return result
