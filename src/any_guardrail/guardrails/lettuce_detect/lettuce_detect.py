import time
from typing import Any, ClassVar

try:
    from lettucedetect.models.inference import HallucinationDetector

    MISSING_PACKAGES_ERROR = None
except ImportError as e:
    MISSING_PACKAGES_ERROR = e

from any_guardrail.base import Guardrail
from any_guardrail.guardrails.utils import default
from any_guardrail.types import CategoryResult, GuardrailOutput, GuardrailUsage, SpanResult


class LettuceDetect(Guardrail):
    """LettuceDetect — token/span-level RAG hallucination detector (KRLabs).

    Wraps the ``lettucedetect`` library to flag spans of an answer that are not
    supported by the provided context. ``validate(input_text, context=..., question=...)``
    treats ``input_text`` as the answer; the returned ``spans`` mark hallucinated text
    (character offsets + confidence ``score``). ``valid`` is ``True`` when no
    hallucinated span is found.

    For more information, see the model cards and library:

    - [lettucedect-base-modernbert-en-v1](https://huggingface.co/KRLabsOrg/lettucedect-base-modernbert-en-v1) (default).
    - [lettucedect-large-modernbert-en-v1](https://huggingface.co/KRLabsOrg/lettucedect-large-modernbert-en-v1).
    - [LettuceDetect on GitHub](https://github.com/KRLabsOrg/LettuceDetect).

    Args:
        model_id: Optional HuggingFace model ID. Defaults to the base ModernBERT model.
        method: Detection method passed to the library. Defaults to ``"transformer"``.

    Raises:
        ImportError: When the ``lettucedetect`` extra is not installed.
    """

    SUPPORTED_MODELS: ClassVar = [
        "KRLabsOrg/lettucedect-base-modernbert-en-v1",
        "KRLabsOrg/lettucedect-large-modernbert-en-v1",
    ]

    def __init__(self, model_id: str | None = None, method: str = "transformer") -> None:
        """Initialize the LettuceDetect guardrail."""
        if MISSING_PACKAGES_ERROR is not None:
            msg = "Missing packages for LettuceDetect guardrail. You can try `pip install 'any-guardrail[lettucedetect]'`"
            raise ImportError(msg) from MISSING_PACKAGES_ERROR
        self.model_id = default(model_id, self.SUPPORTED_MODELS)
        self.detector = HallucinationDetector(method=method, model_path=self.model_id)

    def validate(
        self,
        input_text: str,
        context: str | list[str] | None = None,
        question: str | None = None,
        **kwargs: Any,
    ) -> GuardrailOutput:
        """Detect hallucinated spans in ``input_text`` (the answer) against ``context``.

        Args:
            input_text: The answer to check for hallucinations.
            context: The grounding context (a string or list of strings). Required.
            question: Optional question the answer responds to.
            **kwargs: Ignored.

        Returns:
            A :class:`GuardrailOutput` whose ``spans`` mark hallucinated text;
            ``valid`` is ``True`` when no hallucinated span is found.

        """
        del kwargs
        if context is None:
            msg = "LettuceDetect.validate requires `context` (the grounding text) to check the answer against."
            raise ValueError(msg)
        start = time.perf_counter()
        context_list = [context] if isinstance(context, str) else list(context)
        predictions = self.detector.predict(
            context=context_list,
            question=question or "",
            answer=input_text,
            output_format="spans",
        )
        spans = [
            SpanResult(
                start=int(prediction["start"]),
                end=int(prediction["end"]),
                text=prediction.get("text"),
                label="hallucination",
                score=float(prediction["confidence"]) if prediction.get("confidence") is not None else None,
            )
            for prediction in predictions
        ]
        max_score = max((span.score for span in spans if span.score is not None), default=None)
        result = GuardrailOutput(
            valid=not spans,
            score=max_score,
            categories=[CategoryResult(name="hallucination", triggered=bool(spans))],
            spans=spans or None,
        )
        result.usage = GuardrailUsage(model_id=self.model_id, latency_ms=(time.perf_counter() - start) * 1000.0)
        return result
