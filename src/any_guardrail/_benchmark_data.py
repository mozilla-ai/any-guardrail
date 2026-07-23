"""Hand-authored benchmark results for the model-card registry (issue #194).

Seed committed benchmark numbers here as :class:`~any_guardrail.benchmarks.BenchmarkResult`
entries — each tagged with its provenance (``published:<url>`` or ``measured:<harness-version>``)
and its :class:`~any_guardrail.benchmarks.ComparisonCohort`. The harness and the published-number
harvest fill this from out-of-repo experiment runs; it ships empty so the machinery (registry,
JSON export, JSON Schema, comparability guard, model-card rendering) is in place and tested first.

Example of an entry (do not commit fabricated numbers — this is illustrative):

    BenchmarkResult(
        guardrail="deepset",
        category="prompt_injection",
        value=0.91,
        source=BenchmarkSource(kind="published", url="https://huggingface.co/deepset/deberta-v3-base-injection"),
        cohort=ComparisonCohort(
            dataset="deepset-prompt-injections",
            dataset_revision="test",
            label_mapping="injection=positive",
            metric="f1",
            threshold_policy="f1@0.5",
            harness="published:model-card",
        ),
        contamination=True,  # deepset trained on this dataset
    )
"""

from any_guardrail.benchmarks import BenchmarkResult

# Empty until out-of-repo runs / published-number harvest populate it (see benchmarks/README.md).
BENCHMARK_RESULTS: tuple[BenchmarkResult, ...] = ()
