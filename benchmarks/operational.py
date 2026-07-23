"""Operational measurement for model cards: latency / throughput / memory (issue #194).

Every model card carries operational stats alongside accuracy, following Artificial Analysis's
protocol: p50/p95 end-to-end latency for one request in flight (warmups discarded), throughput at
batch=1, and a memory estimate - all on a **named, pinned hardware tier**. The headline editorial
split to preserve is encoder classifiers (~10-100 ms, <1 GB, ~free self-hosted) vs decoder-LLM
judges (0.5-8 s, 4-20 GB, ~$0.01-0.02/call hosted).

``measure_latency`` below is a real, dependency-free helper. Throughput, memory, and the pinned
hardware description are the pieces that must be captured out-of-repo (they aren't reproducible in
CI) and recorded in the ``ComparisonCohort.harness`` string alongside the numbers.
"""

from __future__ import annotations

import statistics
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


def measure_latency(
    validate: Callable[[str], Any],
    prompts: Sequence[str],
    *,
    warmups: int = 3,
) -> dict[str, float]:
    """Measure p50/p95 end-to-end latency (ms) of a guardrail's ``validate`` over ``prompts``.

    One request in flight, ``warmups`` discarded, following the Artificial Analysis protocol. Pass
    ``guardrail.validate`` as ``validate``. The caller is responsible for pinning and recording the
    hardware tier — latency numbers are meaningless without it.

    Returns ``{"p50_ms", "p95_ms", "mean_ms", "n"}``.
    """
    if not prompts:
        msg = "measure_latency requires at least one prompt"
        raise ValueError(msg)

    for prompt in prompts[:warmups]:
        validate(prompt)

    samples_ms: list[float] = []
    for prompt in prompts:
        start = time.perf_counter()
        validate(prompt)
        samples_ms.append((time.perf_counter() - start) * 1000.0)

    ordered = sorted(samples_ms)
    p95_index = min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))
    return {
        "p50_ms": statistics.median(ordered),
        "p95_ms": ordered[p95_index],
        "mean_ms": statistics.fmean(ordered),
        "n": float(len(ordered)),
    }


def estimate_memory_gb(param_count_b: float, *, four_bit: bool = False) -> float:
    """Estimate fp16 (or 4-bit) memory in GB: ~2 GB/1B params fp16 + ~18% overhead.

    Cross-check against the measured table in arXiv:2502.15427 before publishing; aligns with the
    repo's existing >5 GB ``heavy`` marker.
    """
    bytes_per_param = 0.5 if four_bit else 2.0
    return param_count_b * bytes_per_param * 1.18
