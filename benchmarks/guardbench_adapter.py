"""Adapt any-guardrail guardrails onto the GuardBench moderation-function callback (#194).

GuardBench (EU JRC, EMNLP 2024; EUPL-1.2) evaluates guardrail models over 40 datasets with
standardized F1/AUPRC via a **moderation function**: a callable that takes a batch of conversations
and returns, per item, whether it is unsafe (and optionally a score). This module turns any
``Guardrail`` into that callback, so a single adapter puts every applicable guardrail on GuardBench's
standardized footing (and optionally its public leaderboard).

The ``guardbench`` dependency lives in the ``benchmarks`` dependency-group and is imported lazily,
so this module imports cleanly without it; :func:`evaluate_with_guardbench` raises a helpful error
if it is missing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from any_guardrail.base import Guardrail

# A GuardBench moderation verdict per conversation: unsafe flag + optional risk score.
Verdict = dict[str, Any]


def guardrail_to_moderation_fn(guardrail: Guardrail) -> Any:
    """Return a GuardBench moderation function backed by ``guardrail.validate()``.

    The returned callable maps a batch of conversations (each a string or chat-message list) to a
    list of verdicts ``{"unsafe": bool, "score": float | None}``. ``unsafe`` is ``not
    GuardrailOutput.valid``; ``score`` is the guardrail's canonical risk score (higher = riskier),
    which GuardBench uses for AUPRC when present. This conversion is provider-agnostic and depends
    only on the ``GuardrailOutput`` contract, so it works for every guardrail.
    """

    def moderate(conversations: list[Any]) -> list[Verdict]:
        verdicts: list[Verdict] = []
        for conversation in conversations:
            output = guardrail.validate(conversation)
            verdicts.append({"unsafe": not output.valid, "score": output.score})
        return verdicts

    return moderate


def evaluate_with_guardbench(guardrail: Guardrail, datasets: list[str] | None = None) -> Any:
    """Run GuardBench over ``guardrail`` and return its standardized metrics.

    Thin seam over the GuardBench harness entry point; wire the exact registration/evaluation call
    against the installed ``guardbench`` API during the harness build. Raises ``ImportError`` with an
    install hint when the ``benchmarks`` dependency-group is not installed.
    """
    try:
        import guardbench  # noqa: F401  — presence check; the harness call is wired during the build
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        msg = "GuardBench is not installed. Install the benchmarks group: `uv sync --group benchmarks`."
        raise ImportError(msg) from exc

    moderate = guardrail_to_moderation_fn(guardrail)  # noqa: F841  — the seam the harness build registers
    # TODO(#194): register `moderate` with guardbench and run over `datasets` (defaults to the
    # datasets applicable to this guardrail's categories), returning per-dataset F1/AUPRC keyed by
    # a ComparisonCohort so results drop straight into any_guardrail._benchmark_data.
    msg = "Wire guardbench evaluation here during the harness build; guardrail_to_moderation_fn is ready to register."
    raise NotImplementedError(msg)
