"""Custom benchmark runners for what GuardBench doesn't cover (issue #194).

GuardBench standardizes binary content-safety moderation, but several of our guardrail families
need bespoke evaluation. Each runner below produces benchmark results (tagged
``measured:<harness-version>`` with a comparison cohort) that drop straight into
``any_guardrail._benchmark_data``. They are **stubs**: the datasets are gated / model-heavy, so the
runs happen out-of-repo (see ``benchmarks/README.md``) and the results are committed as JSON.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from any_guardrail.base import Guardrail
    from any_guardrail.benchmarks import BenchmarkResult

_OUT_OF_REPO = (
    "Model-heavy / gated-dataset runner — run out-of-repo per benchmarks/README.md and commit the "
    "resulting BenchmarkResult JSON. See issue #194."
)


def run_judge_pointwise(guardrail: Guardrail) -> list[BenchmarkResult]:
    """Pointwise 1-5 rubric agreement for judge guardrails (FLASK + BiGGen-Bench + Feedback Bench).

    Pearson/Spearman vs reference judgments, evaluated as-shipped. Slots into Prometheus / Selene /
    GLIDER / Flow-Judge; flag Feedback Bench as Prometheus-in-domain.
    """
    raise NotImplementedError(_OUT_OF_REPO)


def run_span_level(guardrail: Guardrail) -> list[BenchmarkResult]:
    """Span-level F1 for span-emitting guardrails (RAGTruth, Mu-SHROOM; PII via presidio-research).

    Exercises LettuceDetect / GliNerPii character spans against human-annotated span benchmarks.
    """
    raise NotImplementedError(_OUT_OF_REPO)


def run_streaming(guardrail: Guardrail) -> list[BenchmarkResult]:
    """Sentence-level streaming eval for Qwen3GuardStream (Qwen3GuardTest ``response_loc`` split).

    Reports on-time-intervention + token-latency, the only human-annotated streaming metric.
    """
    raise NotImplementedError(_OUT_OF_REPO)


def run_pii(guardrail: Guardrail) -> list[BenchmarkResult]:
    """PII span-F1 via ``presidio-research`` scoring (SPY, Gretel PII finance, TAB/ECHR).

    SPY is GliNER2-PII's headline benchmark and fits GliNerPii directly.
    """
    raise NotImplementedError(_OUT_OF_REPO)


RUNNERS: dict[str, Any] = {
    "judge_pointwise": run_judge_pointwise,
    "span_level": run_span_level,
    "streaming": run_streaming,
    "pii": run_pii,
}
