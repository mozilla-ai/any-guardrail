import time
from typing import Any, ClassVar

from any_guardrail.base import Guardrail
from any_guardrail.guardrails.utils import default, spans_from_token_labels
from any_guardrail.providers.base import StandardProvider
from any_guardrail.providers.huggingface import HuggingFaceProvider
from any_guardrail.types import CategoryResult, GuardrailOutput, GuardrailUsage, SpanResult

MAX_LENGTH = 512


class PrivacyFilter(Guardrail):
    """OpenAI Privacy Filter — token-classification PII / secrets detector.

    A token classifier that flags spans of personal data (names, addresses, emails,
    phone numbers, account numbers, secrets, etc.). ``validate`` returns the detected
    character ``spans`` (with entity ``label`` and confidence ``score``); ``valid`` is
    ``True`` when nothing is flagged. Detected entity types are summarized in
    ``categories`` and the riskiest span score is surfaced in ``score``.

    For more information, see the
    [openai/privacy-filter model card](https://huggingface.co/openai/privacy-filter).

    Note: the model ships a custom (sparse-MoE) bidirectional architecture loaded via
    ``trust_remote_code``; span extraction relies on the tokenizer exposing offset
    mappings. Validate on real weights before production use.

    Args:
        model_id: Optional HuggingFace model ID. Defaults to ``openai/privacy-filter``.
        provider: Optional pre-configured provider. Defaults to a ``HuggingFaceProvider``
            loading an ``AutoModelForTokenClassification`` with ``trust_remote_code=True``.
    """

    SUPPORTED_MODELS: ClassVar = ["openai/privacy-filter"]

    def __init__(self, model_id: str | None = None, provider: StandardProvider | None = None) -> None:
        """Initialize the Privacy Filter guardrail."""
        self.model_id = default(model_id, self.SUPPORTED_MODELS)
        if provider is not None:
            self.provider = provider
        else:
            from transformers import AutoModelForTokenClassification, AutoTokenizer

            self.provider = HuggingFaceProvider(
                model_class=AutoModelForTokenClassification,
                tokenizer_class=AutoTokenizer,
                trust_remote_code=True,
            )
        self.provider.load_model(self.model_id)

    def validate(self, input_text: str, **kwargs: Any) -> GuardrailOutput:
        """Detect PII spans in ``input_text``.

        Returns a :class:`GuardrailOutput` whose ``spans`` lists every detected PII span
        (character offsets, entity ``label``, and confidence ``score``). ``valid`` is
        ``True`` when no PII is found.
        """
        del kwargs
        start = time.perf_counter()
        model_inputs = self.provider.pre_process(input_text, return_offsets_mapping=True, truncation=True)
        data = self.provider.infer(model_inputs).data
        spans = spans_from_token_labels(
            data["token_label_ids"][0],
            data["offsets"][0],
            data["id2label"],
            input_text,
            data["token_scores"][0],
        )
        categories = [
            CategoryResult(name=label, triggered=True)
            for label in dict.fromkeys(span.label for span in spans if span.label is not None)
        ]
        max_score = max((span.score for span in spans if span.score is not None), default=None)
        result = GuardrailOutput(
            valid=not spans,
            score=max_score,
            categories=categories,
            spans=spans or None,
        )
        result.usage = GuardrailUsage(model_id=self.model_id, latency_ms=(time.perf_counter() - start) * 1000.0)
        return result

    def _build_spans(self, input_text: str, data: dict[str, Any]) -> list[SpanResult]:
        """Convenience hook re-exposing the span extraction for tests."""
        return spans_from_token_labels(
            data["token_label_ids"][0],
            data["offsets"][0],
            data["id2label"],
            input_text,
            data["token_scores"][0],
        )
