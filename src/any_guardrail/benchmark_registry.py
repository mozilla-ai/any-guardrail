"""Central, import-free registry of guardrail benchmark results (issue #194).

Groups the hand-authored ``any_guardrail._benchmark_data`` by guardrail and exposes the
comparability guard the model-card renderer relies on. Imports only ``any_guardrail.base``
(for ``GuardrailName``), ``any_guardrail.benchmarks`` (leaf models), and the
``any_guardrail._benchmark_data`` leaf — never a guardrail implementation — so reading benchmark
results never pulls in ``torch`` / ``transformers`` or spins up a backend.
"""

from any_guardrail._benchmark_data import BENCHMARK_RESULTS
from any_guardrail.base import GuardrailName
from any_guardrail.benchmarks import BenchmarkResult, ComparisonCohort


def _by_guardrail() -> dict[GuardrailName, tuple[BenchmarkResult, ...]]:
    grouped: dict[GuardrailName, list[BenchmarkResult]] = {name: [] for name in GuardrailName}
    for result in BENCHMARK_RESULTS:
        grouped[GuardrailName(result.guardrail)].append(result)
    return {name: tuple(results) for name, results in grouped.items()}


BENCHMARK_REGISTRY: dict[GuardrailName, tuple[BenchmarkResult, ...]] = _by_guardrail()


def get_benchmarks(name: GuardrailName) -> list[BenchmarkResult]:
    """Return the committed benchmark results for a guardrail (empty list when none)."""
    return list(BENCHMARK_REGISTRY[name])


def group_comparable(results: list[BenchmarkResult]) -> dict[ComparisonCohort, list[BenchmarkResult]]:
    """Partition results into comparable cohorts.

    Two results share a group **iff** their :class:`ComparisonCohort` is equal, i.e. their dataset
    revision, label mapping, metric, threshold policy, harness, etc. all match. Results in
    different cohorts must never be ranked against each other — the renderer shows each cohort in
    its own table and never aligns scores across cohorts, so the comparability traps in the
    guard-model literature cannot silently produce a misleading leaderboard. Insertion order within
    a cohort is preserved.
    """
    groups: dict[ComparisonCohort, list[BenchmarkResult]] = {}
    for result in results:
        groups.setdefault(result.cohort, []).append(result)
    return groups
