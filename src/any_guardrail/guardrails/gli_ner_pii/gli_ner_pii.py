import time
from typing import Any, ClassVar

from any_guardrail.base import GuardrailName
from any_guardrail.registry import GUARDRAIL_METADATA
from any_guardrail.taxonomy import GuardrailMetadata

try:
    from gliner2 import GLiNER2

    MISSING_PACKAGES_ERROR = None
except ImportError as e:
    MISSING_PACKAGES_ERROR = e

from any_guardrail.base import Guardrail
from any_guardrail.guardrails.utils import default
from any_guardrail.types import CategoryResult, GuardrailOutput, GuardrailUsage, SpanResult

# A broad default PII taxonomy covering the model's semantic groups (names, contact,
# government/tax IDs, banking, digital identity, secrets). GLiNER2 is zero-shot, so callers
# can override this per call via ``entity_types=`` to widen, narrow, or rename the labels.
DEFAULT_PII_ENTITIES = [
    "person",
    "email",
    "phone_number",
    "address",
    "credit_card_number",
    "social_security_number",
    "date_of_birth",
    "ip_address",
    "passport_number",
    "driver_license_number",
    "bank_account_number",
    "api_key",
    "password",
]


class GliNerPii(Guardrail):
    """Span-level PII/NER detector emitting character spans and a redacted copy of the text.

    Wraps the ``gliner2`` library to run a GLiNER2 encoder that extracts personally
    identifiable information as character-offset spans, in a single pass, for a
    caller-supplied (or default) list of entity types. Because GLiNER2 is zero-shot, the
    ``entity_types`` are specified at call time (defaulting to ``DEFAULT_PII_ENTITIES``),
    so the same model can detect an arbitrary PII taxonomy without retraining.

    Verdict mapping onto ``GuardrailOutput``:

    - ``spans`` marks each detected entity (character ``start`` / ``end`` offsets into the
      input, the offending ``text``, ``label`` = the entity type, and a confidence ``score``).
    - ``valid`` is ``True`` when no PII span is found.
    - ``score`` is the maximum span confidence (higher = riskier; ``None`` when nothing is flagged).
    - ``categories`` holds one entry per detected entity type (``triggered=True``).
    - ``modified_text`` is a redacted copy of the input with each detected span replaced by
      ``redaction_placeholder`` (``None`` when nothing is flagged), so a downstream redact
      mitigation can consume it directly.

    Expected inputs: a single text string. Extra keyword arguments to ``validate`` are ignored.

    This guardrail wraps an upstream library rather than a provider, so it requires the
    ``gliner`` extra (``pip install 'any-guardrail[gliner]'``); the constructor re-raises a
    helpful ``ImportError`` when it is missing.

    For more information, see:

    - [gliner2-privacy-filter-PII-multi model card](https://huggingface.co/fastino/gliner2-privacy-filter-PII-multi) (default).
    - [GLiNER2-PII: A Multilingual Model for Personally Identifiable Information Extraction
      (arXiv:2605.09973)](https://arxiv.org/abs/2605.09973).

    Args:
        model_id: Optional HuggingFace model ID from ``SUPPORTED_MODELS``. Defaults to
            ``fastino/gliner2-privacy-filter-PII-multi``.
        threshold: Default confidence threshold passed to ``gliner2``'s ``extract_entities``
            (a per-call ``threshold`` overrides it). Defaults to 0.5.

    Raises:
        ImportError: When the ``gliner`` extra is not installed.

    """

    SUPPORTED_MODELS: ClassVar = ["fastino/gliner2-privacy-filter-PII-multi"]

    METADATA: ClassVar[GuardrailMetadata] = GUARDRAIL_METADATA[GuardrailName.GLI_NER_PII]

    def __init__(self, model_id: str | None = None, threshold: float = 0.5) -> None:
        """Initialize the GLiNER2 PII guardrail.

        Args:
            model_id: Optional HuggingFace model ID. Must be one of ``SUPPORTED_MODELS``;
                defaults to ``fastino/gliner2-privacy-filter-PII-multi``.
            threshold: Default confidence threshold forwarded to ``gliner2``'s
                ``extract_entities``. Defaults to 0.5; a per-call ``threshold`` overrides it.

        Raises:
            ImportError: When the ``gliner`` extra is not installed.
            ValueError: If ``model_id`` is not in ``SUPPORTED_MODELS``.

        """
        if MISSING_PACKAGES_ERROR is not None:
            msg = "Missing packages for GliNerPii guardrail. You can try `pip install 'any-guardrail[gliner]'`"
            raise ImportError(msg) from MISSING_PACKAGES_ERROR
        self.model_id = default(model_id, self.SUPPORTED_MODELS)
        self.threshold = threshold
        self.model = GLiNER2.from_pretrained(self.model_id)

    def validate(
        self,
        input_text: str,
        entity_types: list[str] | None = None,
        threshold: float | None = None,
        redaction_placeholder: str = "[REDACTED_{label}]",
        **kwargs: Any,
    ) -> GuardrailOutput:
        """Detect PII spans in ``input_text`` and emit a redacted copy.

        Args:
            input_text: The text to scan for PII. A single string.
            entity_types: PII entity types to detect. Defaults to ``DEFAULT_PII_ENTITIES``.
            threshold: Confidence threshold for this call. Defaults to the instance ``threshold``.
            redaction_placeholder: Template used to replace each detected span in
                ``modified_text``; the substring ``{label}`` is replaced with the upper-cased
                entity type (e.g. ``"[REDACTED_EMAIL]"``). Any other braces are kept literal, so
                an arbitrary placeholder never raises.
            **kwargs: Accepted for interface compatibility and ignored.

        Returns:
            A :class:`GuardrailOutput` whose ``spans`` mark detected PII, ``valid`` is ``True``
            when none is found, and ``modified_text`` is a redacted copy of the input.

        """
        del kwargs
        start = time.perf_counter()
        labels = entity_types if entity_types is not None else DEFAULT_PII_ENTITIES
        used_threshold = threshold if threshold is not None else self.threshold
        result: dict[str, Any] = self.model.extract_entities(
            input_text,
            labels,
            threshold=used_threshold,
            include_spans=True,
            include_confidence=True,
        )

        spans = _to_spans(result)
        triggered_labels = sorted({span.label for span in spans if span.label})
        max_score = max((span.score for span in spans if span.score is not None), default=None)
        out = GuardrailOutput(
            valid=not spans,
            score=max_score,
            categories=[CategoryResult(name=label, triggered=True) for label in triggered_labels],
            spans=spans or None,
            modified_text=_redact(input_text, spans, redaction_placeholder) if spans else None,
        )
        out.usage = GuardrailUsage(model_id=self.model_id, latency_ms=(time.perf_counter() - start) * 1000.0)
        return out


def _to_spans(result: dict[str, Any]) -> list[SpanResult]:
    """Flatten a gliner2 ``extract_entities`` result into ``SpanResult``s (sorted by offset).

    The result maps each entity type to a list of ``{"text", "confidence", "start", "end"}``
    dicts (from ``include_spans=True, include_confidence=True``). Entries without character
    offsets are skipped, since a span with no offsets can't be located.
    """
    entities = result.get("entities", result)
    spans: list[SpanResult] = []
    for label, items in entities.items():
        for item in items if isinstance(items, list) else [items]:
            if not isinstance(item, dict) or item.get("start") is None or item.get("end") is None:
                continue
            confidence = item.get("confidence")
            spans.append(
                SpanResult(
                    start=int(item["start"]),
                    end=int(item["end"]),
                    text=item.get("text"),
                    label=str(label),
                    score=float(confidence) if confidence is not None else None,
                )
            )
    spans.sort(key=lambda span: (span.start, span.end))
    return spans


def _redact(text: str, spans: list[SpanResult], placeholder: str) -> str:
    """Return ``text`` with every detected span replaced by ``placeholder``.

    Overlapping spans (possible across entity types) are merged into their maximal covering
    region so the **whole** flagged region is redacted — a shorter high-confidence span nested
    in a longer one never leaves the longer span's tail in the clear. Each merged region is
    labelled by its highest-scoring span.
    """
    ordered = sorted(spans, key=lambda span: (span.start, span.end))
    # Merge overlapping spans into [start, end, label, best_score] regions covering their union.
    regions: list[list[Any]] = []
    for span in ordered:
        score = span.score or 0.0
        if regions and span.start < regions[-1][1]:
            region = regions[-1]
            region[1] = max(region[1], span.end)
            if score > region[3]:
                region[2], region[3] = span.label, score
        else:
            regions.append([span.start, span.end, span.label, score])
    pieces: list[str] = []
    cursor = 0
    for start, end, label, _ in regions:
        pieces.append(text[cursor:start])
        pieces.append(_format_placeholder(placeholder, label))
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces)


def _format_placeholder(placeholder: str, label: str | None) -> str:
    """Substitute ``{label}`` (upper-cased) into ``placeholder`` without ``str.format``.

    A plain ``replace`` leaves any other braces in a user-supplied placeholder literal, so an
    arbitrary ``redaction_placeholder`` can never raise ``KeyError`` / ``ValueError``.
    """
    return placeholder.replace("{label}", (label or "PII").upper())
