# Benchmark harness (issue #194)

Model-heavy evaluation machinery that produces the numbers rendered on the guardrail model-card
pages. **Unshipped** (outside `src/`, never in the wheel) and **never run in CI** — runs happen
out-of-repo per release / when a guardrail lands, and results are committed as JSON.

## Layout

| File | Role |
|---|---|
| `guardbench_adapter.py` | `Guardrail.validate()` → GuardBench moderation-function callback (the standardized footing for content-safety / prompt-injection). |
| `runners.py` | Small custom runners for what GuardBench lacks: judge pointwise suite, span/streaming evals, PII scoring. **Stubs** — filled in out-of-repo. |
| `operational.py` | Latency (p50/p95), throughput, memory on **pinned hardware**. `measure_latency` is real; hardware pinning is the caller's job. |

Results land in `src/any_guardrail/_benchmark_data.py` as typed
`BenchmarkResult(...)` entries and are exported to `schemas/guardrail_benchmarks.json`
(`python scripts/generate_benchmarks_json.py`). The model-card generator
(`scripts/generate_api_docs.py`) renders them per guardrail.

## The comparability invariant (non-negotiable)

Every score carries a `ComparisonCohort` (dataset + immutable revision/split, label mapping + positive
class, metric + implementation, threshold policy, harness/prompt/rubric version, aggregation level).
The renderer groups by cohort and **never aligns scores across cohorts** — so the traps the guard-model
literature is full of cannot silently produce a misleading leaderboard:

- ToxicChat **1123 vs 0124** → different `dataset_revision` → separate cohorts.
- **Optimal-F1 vs F1@0.5** → different `threshold_policy` → separate cohorts.
- **AUC dropped into an F1 column** → different `metric` → separate cohorts.
- a **baseline copied from another paper's harness** → different `harness` → separate cohorts.
- a **missing** result is `value=None` and is rendered as `—`, **never `0`**.
- every number carries provenance (`published:<url>` or `measured:<harness-version>`); a source is required.

`tests/unit/test_benchmarks.py` encodes these as adversarial fixtures.

## Principles (from the issue)

1. **Recycle first, compute second.** Harvest published numbers (tagged `published:<url>`) before running anything; the widest columns are ToxicChat / OpenAI-Moderation (content safety) and the InjecGuard/PIGuard NotInject+BIPIA+PINT table (prompt injection).
2. **Suites keyed by `GuardrailCategory`**; test-split-only; contamination flags on any in-distribution cell.
3. **Always pair detection with an over-defense / FPR benchmark** (e.g. XSTest, NotInject).
4. **Judges evaluated as-shipped** — pin the exact prompt/rubric (the prompt registry, #20/#87) in the cohort.

## Per-dataset license / access (do the one-time NC legal check before first publication)

Publishing an aggregate *score* is publishing an uncopyrightable fact (no CC clause attaches). The only
gray zone is the local-download step for NC-licensed sets — mitigated by their being released *as eval
benchmarks* and by universal community practice. Keep this table current as datasets are added:

| Dataset | License | Access | Category |
|---|---|---|---|
| NotInject | MIT | open | prompt injection (over-defense) |
| deepset test split | Apache-2.0 | open | prompt injection |
| ToxicChat | CC-BY-NC | download (NC — note) | content safety |
| OpenAI Moderation eval | MIT | open | content safety |
| AEGIS 2.0 test | CC-BY-4.0 | open | content safety |
| XSTest | CC-BY-4.0 | open | over-refusal FPR |
| WildGuardTest | ODC-BY | **gated** (HF) | content safety |
| RAGTruth test | MIT | open | hallucination (span) |
| Mu-SHROOM | CC-BY | open | hallucination (span) |
| LLM-AggreFact | click-through permits eval | gated | hallucination |
| SPY | CC-BY | open | PII (span) |
| PolyGuardPrompts | CC-BY-4.0 | open | multilingual |
| RTP-LX | MIT | open | multilingual (benign+toxic) |
| GovTech off-topic 17.2k | MIT | open | off-topic |

## Running out-of-repo

```bash
uv sync --group benchmarks        # pulls guardbench + presidio-analyzer (not in the wheel, not in CI)
# then, on a pinned hardware tier, drive the adapter/runners and commit the resulting
# BenchmarkResult JSON into src/any_guardrail/_benchmark_data.py + regenerate the schemas.
```

Gated datasets need `HF_TOKEN` + accepting the dataset terms; the harness documents the access step
rather than auto-downloading. Optionally publish runs to the GuardBench HF leaderboard.
