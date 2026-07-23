"""Benchmark-result models for the guardrail model-card benchmark registry (issue #194).

These frozen, dependency-free models let the repo commit **benchmark numbers** — measured with
the harness or recycled from a paper/model card — next to each guardrail, so generated model-card
pages can pair the #182 capability metadata with evidence.

The load-bearing idea is **comparability as a machine-enforced invariant**, not a footnote: every
score carries a :class:`ComparisonCohort` capturing exactly the keys that must match for two
scores to be directly comparable (dataset revision, label mapping, metric + threshold policy, and
the harness/prompt that produced it). The registry's ``group_comparable`` refuses to place scores
from different cohorts in the same ranked table, so the traps the literature is full of — ToxicChat
1123 vs 0124, Optimal-F1 vs F1@0.5, an AUC dropped into an F1 column, a baseline copied from a
different harness — cannot silently produce a misleading leaderboard. A missing score is ``None``
and must never be coerced to ``0``.

This module is deliberately dependency-free (only the standard library and Pydantic).
"""

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator


class BenchmarkSourceKind(StrEnum):
    """Where a benchmark number came from."""

    PUBLISHED = "published"
    """Recycled from a model card / paper (carries the source ``url``)."""

    MEASURED = "measured"
    """Computed by the benchmark harness (carries the ``harness_version``)."""


class BenchmarkSource(BaseModel):
    """Provenance of a single benchmark number: ``published:<url>`` or ``measured:<harness>``."""

    model_config = ConfigDict(frozen=True)

    kind: BenchmarkSourceKind
    """Whether the number was published elsewhere or measured here."""

    url: str | None = None
    """Source URL for a ``published`` number (model card, paper, or leaderboard)."""

    harness_version: str | None = None
    """Harness version string for a ``measured`` number (pins reproducibility)."""

    @model_validator(mode="after")
    def _require_provenance(self) -> Self:
        """Require a URL for a published number and a harness version for a measured one."""
        if self.kind is BenchmarkSourceKind.PUBLISHED and not self.url:
            msg = "a published BenchmarkSource requires a `url`"
            raise ValueError(msg)
        if self.kind is BenchmarkSourceKind.MEASURED and not self.harness_version:
            msg = "a measured BenchmarkSource requires a `harness_version`"
            raise ValueError(msg)
        return self


class ComparisonCohort(BaseModel):
    """The keys that must all match for two scores to be directly comparable / rankable (#194).

    Two benchmark results are comparable **iff** their cohorts are equal. The renderer groups by
    cohort and never aligns scores across cohorts, which is what makes the comparability traps in
    the guard-model literature impossible to render as one ranked column.
    """

    model_config = ConfigDict(frozen=True)

    dataset: str
    """Benchmark dataset name (e.g. ``"ToxicChat"``, ``"NotInject"``)."""

    dataset_revision: str
    """Immutable revision / split hash / version (e.g. ToxicChat ``"1123"`` vs ``"0124"``)."""

    split: str = "test"
    """Evaluation split (test-split-only by convention)."""

    label_mapping: str
    """Label mapping + positive class identity (which label counts as "flagged")."""

    metric: str
    """Metric name (e.g. ``"f1"``, ``"auprc"``, ``"auc"``)."""

    metric_impl: str | None = None
    """Metric implementation / version, when it affects the number."""

    threshold_policy: str
    """Threshold policy (e.g. ``"f1@0.5"``, ``"optimal-f1"``, ``"n/a"`` for threshold-free metrics)."""

    harness: str
    """Harness / prompt / rubric version that produced the number (re-run baselines differ)."""

    aggregation: str = "dataset"
    """Aggregation level (e.g. ``"dataset"``, ``"macro"``, ``"per-language"``)."""


class BenchmarkResult(BaseModel):
    """One guardrail's score on one benchmark under one comparison cohort."""

    model_config = ConfigDict(frozen=True)

    guardrail: str
    """The guardrail this score is for (a :class:`~any_guardrail.base.GuardrailName` value)."""

    category: str
    """The suite section (a :class:`~any_guardrail.taxonomy.GuardrailCategory` value)."""

    value: float | None
    """The score. ``None`` means **missing** and must never be rendered or ranked as ``0``."""

    source: BenchmarkSource
    """Provenance (published/measured)."""

    cohort: ComparisonCohort
    """The comparability keys; only same-cohort scores may be ranked together."""

    contamination: bool = False
    """``True`` when the guardrail trained on (or is in-distribution with) this benchmark."""

    notes: str | None = None
    """Optional free-text caveat (rendered as a footnote)."""
