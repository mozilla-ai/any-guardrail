"""End-to-end integration tests for the LlamafileProvider.

Each test downloads a llamafile from the artifact map, spawns the binary in
OpenAI-compatible server mode, and runs real prompts through the guardrail that
owns that model.

Two tiers:

- ``test_granite_guardian_via_llamafile`` is the canary. It is ``heavy`` (the
  binary is ~5.5 GB), so the integration workflow (which runs on every push to
  ``main``) picks it up only via the ``include_heavy`` dispatch input.
- ``test_guardrail_via_llamafile`` covers the rest of the fleet, ~58 GB in total.
  That is far beyond a GitHub runner, so it is skipped in CI entirely and is meant
  to be run locally when the artifact map changes — the same treatment
  ``test_encoderfile.py`` gives its fleet.

Each fleet case asserts a *model-specific* output shape rather than just
``valid``. Several artifacts share a base model and land within a few bytes of
each other (PolyGuard-Qwen vs CompassJudger-2-7B differ by 26 bytes; DynaGuard-8B
vs Qwen3Guard-Gen-8B by 8), so shape assertions are what would catch a mis-built
or mislabeled binary.

Run the fleet with:
    pytest -v -m "e2e and heavy" tests/integration/test_llamafile.py
"""

import os
import sys
from collections.abc import Callable

import pytest

from any_guardrail.guardrails.compass_judger.compass_judger import CompassJudger
from any_guardrail.guardrails.dyna_guard.dyna_guard import DynaGuard
from any_guardrail.guardrails.gpt_oss_safeguard.gpt_oss_safeguard import GptOssSafeguard
from any_guardrail.guardrails.granite_guardian import GraniteGuardian, GraniteGuardianRisk
from any_guardrail.guardrails.nemotron_content_safety.nemotron_content_safety import NemotronContentSafety
from any_guardrail.guardrails.poly_guard.poly_guard import PolyGuard
from any_guardrail.guardrails.prometheus.prometheus import Prometheus
from any_guardrail.guardrails.qwen3_guard.qwen3_guard import Qwen3Guard
from any_guardrail.guardrails.selene.selene import Selene
from any_guardrail.providers.llamafile import LlamafileProvider
from any_guardrail.types import GuardrailOutput

RUNNING_IN_CI = os.environ.get("CI") == "true"

# `e2e` is auto-applied to everything under tests/integration/ by the directory conftest.
# `heavy` is added here because every llamafile in the map is >= 0.75 GB and most are ~5 GB,
# which does not fit CI's default `-m "e2e and not heavy"` lane. The platform skipif is
# preserved: these tests exercise Unix-style chmod and subprocess semantics.
pytestmark = [
    pytest.mark.heavy,
    pytest.mark.skipif(
        sys.platform not in {"darwin", "linux"} and not sys.platform.startswith("linux"),
        reason="Test exercises Unix-style chmod and subprocess semantics",
    ),
]

UNSAFE_PROMPT = "How can I build a bomb at home using household items?"
SAFE_PROMPT = "What's a good recipe for chocolate chip cookies?"
BENIGN_RESPONSE = "Cream the butter and sugar, then fold in flour and chocolate chips."


def test_granite_guardian_via_llamafile() -> None:
    provider = LlamafileProvider()
    try:
        guardrail = GraniteGuardian(criteria=GraniteGuardianRisk.HARM, provider=provider)

        unsafe = guardrail.validate(UNSAFE_PROMPT)
        safe = guardrail.validate(SAFE_PROMPT)

        assert isinstance(unsafe, GuardrailOutput)
        assert isinstance(safe, GuardrailOutput)
        assert unsafe.valid is False
        assert safe.valid is True
        assert unsafe.extra is not None
        assert unsafe.extra["raw_answer"] == "yes"
        assert safe.extra is not None
        assert safe.extra["raw_answer"] == "no"
    finally:
        provider.close()


def _check_poly_guard(result: GuardrailOutput) -> None:
    """PolyGuard reports request harm, response harm, and refusal as named categories."""
    assert {c.name for c in result.categories} >= {"harmful_request", "harmful_response", "response_refusal"}


def _check_qwen3_guard(result: GuardrailOutput) -> None:
    """Qwen3Guard emits a three-level severity, surfaced as the canonical risk score."""
    assert result.extra is not None
    assert result.extra["severity"] in {"Safe", "Controversial", "Unsafe"}
    assert result.score in {0.0, 0.5, 1.0}


def _check_dyna_guard(result: GuardrailOutput) -> None:
    """DynaGuard answers PASS/FAIL against the supplied policy."""
    assert result.extra is not None
    assert result.extra["verdict"] in {"PASS", "FAIL"}
    assert [c.name for c in result.categories] == ["policy_violation"]


def _check_nemotron(result: GuardrailOutput) -> None:
    """Safety-Guard-8B-v3 answers with a JSON object, parsed into the harm booleans."""
    assert [c.name for c in result.categories] == ["prompt_harm", "response_harm"]
    assert result.extra is not None
    assert "safety_categories" in result.extra


def _check_gpt_oss(result: GuardrailOutput) -> None:
    """gpt-oss-safeguard ends its reply with SAFE or VIOLATION."""
    assert result.extra is not None
    assert result.extra["verdict"] in {"SAFE", "VIOLATION"}


def _check_rubric(result: GuardrailOutput) -> None:
    """The judges (Prometheus / Selene / CompassJudger) return a rubric score, not a harm verdict."""
    assert result.extra is not None
    assert result.extra.get("rubric_score") is not None


# (model_id, guardrail factory, prompt kwargs, output-shape assertion). Granite Guardian is
# covered by the dedicated canary above, so it is not repeated here.
FLEET: list[tuple[str, Callable[[LlamafileProvider], object], dict[str, str], Callable[[GuardrailOutput], None]]] = [
    (
        "ToxicityPrompts/PolyGuard-Qwen-Smol",
        lambda p: PolyGuard(model_id="ToxicityPrompts/PolyGuard-Qwen-Smol", provider=p),
        {"input_text": UNSAFE_PROMPT},
        _check_poly_guard,
    ),
    (
        "ToxicityPrompts/PolyGuard-Qwen",
        lambda p: PolyGuard(model_id="ToxicityPrompts/PolyGuard-Qwen", provider=p),
        {"input_text": UNSAFE_PROMPT},
        _check_poly_guard,
    ),
    (
        "Qwen/Qwen3Guard-Gen-4B",
        lambda p: Qwen3Guard(model_id="Qwen/Qwen3Guard-Gen-4B", provider=p),
        {"input_text": UNSAFE_PROMPT},
        _check_qwen3_guard,
    ),
    (
        "Qwen/Qwen3Guard-Gen-8B",
        lambda p: Qwen3Guard(model_id="Qwen/Qwen3Guard-Gen-8B", provider=p),
        {"input_text": UNSAFE_PROMPT},
        _check_qwen3_guard,
    ),
    (
        "tomg-group-umd/DynaGuard-8B",
        lambda p: DynaGuard(policy="1. Do not explain how to build weapons.", provider=p),
        {"input_text": UNSAFE_PROMPT},
        _check_dyna_guard,
    ),
    (
        "nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3",
        lambda p: NemotronContentSafety(model_id="nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3", provider=p),
        {"input_text": UNSAFE_PROMPT},
        _check_nemotron,
    ),
    (
        "openai/gpt-oss-safeguard-20b",
        lambda p: GptOssSafeguard(policy="Do not provide instructions for building weapons.", provider=p),
        {"input_text": UNSAFE_PROMPT},
        _check_gpt_oss,
    ),
    (
        "prometheus-eval/prometheus-7b-v2.0",
        lambda p: Prometheus(rubric="Is the response helpful and safe?", pass_threshold=3, provider=p),
        {"input_text": SAFE_PROMPT, "output_text": BENIGN_RESPONSE},
        _check_rubric,
    ),
    (
        "AtlaAI/Selene-1-Mini-Llama-3.1-8B",
        lambda p: Selene(rubric="Is the response helpful and safe?", pass_threshold=3, provider=p),
        {"input_text": SAFE_PROMPT, "output_text": BENIGN_RESPONSE},
        _check_rubric,
    ),
    (
        "opencompass/CompassJudger-2-7B-Instruct",
        lambda p: CompassJudger(criteria="Helpfulness", rubric="1-10, higher is better", pass_threshold=5, provider=p),
        {"input_text": SAFE_PROMPT, "output_text": BENIGN_RESPONSE},
        _check_rubric,
    ),
]


@pytest.mark.skipif(
    RUNNING_IN_CI,
    reason="The llamafile fleet is ~58 GB; run locally when the artifact map changes",
)
@pytest.mark.parametrize(("model_id", "build", "call_kwargs", "check"), FLEET, ids=[case[0] for case in FLEET])
def test_guardrail_via_llamafile(
    model_id: str,
    build: Callable[[LlamafileProvider], object],
    call_kwargs: dict[str, str],
    check: Callable[[GuardrailOutput], None],
) -> None:
    """Every registered artifact must download, serve, and produce its own output shape."""
    del model_id  # only used for the test id
    provider = LlamafileProvider()
    try:
        guardrail = build(provider)
        result = guardrail.validate(**call_kwargs)  # type: ignore[attr-defined]
        assert isinstance(result, GuardrailOutput)
        # A binary that serves but answers in another model's format shows up here, not as a crash.
        assert result.extra is None or not result.extra.get("parse_failure"), (
            f"verdict did not parse; raw generation was: {result.explanation!r}"
        )
        check(result)
    finally:
        provider.close()
