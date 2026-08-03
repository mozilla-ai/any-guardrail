"""Benchmark harness for any-guardrail model cards (issue #194).

This is a **top-level, unshipped** package (it lives beside ``scripts/`` and ``schemas/``, outside
``src/``, so it is never bundled in the wheel). It holds the model-heavy evaluation machinery that
produces the numbers committed to ``any_guardrail._benchmark_data`` / ``schemas/guardrail_benchmarks.json``:

- :mod:`benchmarks.guardbench_adapter` — maps ``Guardrail.validate()`` onto GuardBench's
  moderation-function callback, putting every applicable guardrail on a standardized footing.
- :mod:`benchmarks.runners` — small custom runners for what GuardBench lacks (judge pointwise
  suite, span/streaming evals, PII scoring). Stubs today; filled in per the roadmap in ``README.md``.
- :mod:`benchmarks.operational` — latency / throughput / memory measurement on pinned hardware.

Nothing here runs in CI. Heavy dependencies (``guardbench``, ``presidio-analyzer``) live in the
``benchmarks`` dependency-group and are imported lazily behind ``try/except ImportError`` so the
package imports cleanly without them. See ``benchmarks/README.md`` for the methodology, the
per-dataset license/access table, and the comparability invariant.
"""
