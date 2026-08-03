# Benchmarks & Model Cards

Each guardrail's API page carries a **Benchmarks** section pairing its capability metadata with
*measured* or *recycled* benchmark numbers, and a **License** section — so you can pick between
guardrails on evidence, not descriptions.

The numbers live in an import-free registry (`any_guardrail.benchmark_registry`), exported to
[`schemas/guardrail_benchmarks.json`](https://github.com/mozilla-ai/any-guardrail/blob/main/schemas/guardrail_benchmarks.json)
and validated against
[`schemas/guardrail_benchmarks.schema.json`](https://github.com/mozilla-ai/any-guardrail/blob/main/schemas/guardrail_benchmarks.schema.json).
The model-heavy harness that produces them lives in the unshipped
[`benchmarks/`](https://github.com/mozilla-ai/any-guardrail/tree/main/benchmarks) package and never
runs in CI.

## Comparability is machine-enforced

Every score carries a `ComparisonCohort`. Two scores are comparable **only if** their cohort is
equal — same dataset revision, label mapping, metric, threshold policy, and harness. The renderer
groups by cohort and never aligns scores across cohorts, so ToxicChat 1123 vs 0124, Optimal-F1 vs
F1@0.5, or an AUC in an F1 column can't silently become one ranked column. A missing score is
`None` and renders as `—`, never `0`; every number carries provenance.

```python
from any_guardrail import BenchmarkResult, BenchmarkSource, ComparisonCohort

result = BenchmarkResult(
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
assert result.value == 0.91
assert result.source.kind == "published"
```

## Adding numbers

Harvest published numbers (tag `published:<url>`) or run the harness on pinned hardware
(tag `measured:<harness-version>`), append `BenchmarkResult(...)` entries to
`src/any_guardrail/_benchmark_data.py`, then regenerate:

```
python scripts/generate_benchmarks_json.py
python scripts/generate_benchmark_schema.py
```

See [`benchmarks/README.md`](https://github.com/mozilla-ai/any-guardrail/blob/main/benchmarks/README.md)
for the methodology, per-dataset license/access table, and the one-time legal check on NC datasets.
