"""Post-processing / parse-logic tests for the issue #179 guardrails.

Instances are built with ``object.__new__`` so no model weights load; only the
parsing of provider output into ``GuardrailOutput`` is exercised.
"""

from typing import Any

import pytest

from any_guardrail.guardrails.bielik_guard.bielik_guard import _build_output as bielik_build
from any_guardrail.guardrails.compass_judger.compass_judger import CompassJudger
from any_guardrail.guardrails.dyna_guard.dyna_guard import DynaGuard
from any_guardrail.guardrails.gpt_oss_safeguard.gpt_oss_safeguard import GptOssSafeguard
from any_guardrail.guardrails.kanana_safeguard.kanana_safeguard import KANANA_CATEGORIES, KananaSafeguard
from any_guardrail.guardrails.nemotron_content_safety.nemotron_content_safety import NemotronContentSafety
from any_guardrail.guardrails.poly_guard.poly_guard import PolyGuard
from any_guardrail.guardrails.prometheus.prometheus import Prometheus
from any_guardrail.guardrails.prompt_guard.prompt_guard import _build_output as prompt_build
from any_guardrail.guardrails.qwen3_guard.qwen3_guard import Qwen3Guard
from any_guardrail.guardrails.qwen3_guard_stream.qwen3_guard_stream import Qwen3GuardStream
from any_guardrail.guardrails.selene.selene import Selene
from any_guardrail.guardrails.wild_guard.wild_guard import WildGuard
from any_guardrail.types import GuardrailInferenceOutput, GuardrailOutput


def _gen(text: str) -> GuardrailInferenceOutput[Any]:
    """Mimic a provider.generate_chat output."""
    return GuardrailInferenceOutput(data={"generated_text": text, "prompt_token_count": 7, "completion_token_count": 3})


# --- Encoder classifiers -------------------------------------------------------


def test_prompt_guard_flags_malicious_by_index() -> None:
    benign = prompt_build([0.9, 0.1], 0, ["benign", "malicious"])
    malicious = prompt_build([0.2, 0.8], 1, ["benign", "malicious"])
    assert benign.valid is True
    assert benign.score == pytest.approx(0.1)
    assert malicious.valid is False
    assert malicious.score == pytest.approx(0.8)


def test_bielik_multilabel_thresholding() -> None:
    labels = ["HATE", "VULGAR", "SEX", "CRIME", "SELF_HARM"]
    result = bielik_build([0.1, 0.2, 0.9, 0.05, 0.05], labels, 0.5)
    assert result.valid is False
    assert result.score == pytest.approx(0.9)
    triggered = [c.name for c in result.categories if c.triggered]
    assert triggered == ["SEX"]


def test_bielik_all_below_threshold_is_valid() -> None:
    result = bielik_build([0.1, 0.2, 0.3], ["A", "B", "C"], 0.5)
    assert result.valid is True


# --- Decoder safety classifiers ------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected_valid"),
    [
        ("Harmful request: no\nResponse refusal: no\nHarmful response: no", True),
        ("Harmful request: yes\nResponse refusal: no\nHarmful response: no", False),
        ("Harmful request: no\nResponse refusal: yes\nHarmful response: yes", False),
    ],
)
def test_wildguard_parse(text: str, expected_valid: bool) -> None:
    instance = object.__new__(WildGuard)
    result = instance._post_processing(_gen(text))
    assert result.valid is expected_valid


def test_wildguard_fails_closed() -> None:
    instance = object.__new__(WildGuard)
    result = instance._post_processing(_gen("garbage with no fields"))
    assert result.valid is False
    assert result.extra == {"parse_failure": True}


def test_polyguard_parse_with_categories() -> None:
    instance = object.__new__(PolyGuard)
    text = "Harmful request: yes\nRequest safety violations: S1,S10\nResponse refusal: no\nHarmful response: no"
    result = instance._post_processing(_gen(text))
    assert result.valid is False
    names = {c.name for c in result.categories if c.triggered}
    assert "S1" in names
    assert "S10" in names


def test_polyguard_fails_closed() -> None:
    instance = object.__new__(PolyGuard)
    result = instance._post_processing(_gen("nope"))
    assert result.valid is False
    assert result.extra == {"parse_failure": True}


@pytest.mark.parametrize(
    ("text", "expected_valid"),
    [
        ("<answer>PASS</answer>", True),
        ("<answer>FAIL</answer>", False),
        ("<think>reasoning</think>\n<answer>FAIL</answer>", False),
        ("The agent complied. PASS", True),
    ],
)
def test_dynaguard_parse(text: str, expected_valid: bool) -> None:
    instance = object.__new__(DynaGuard)
    result = instance._post_processing(_gen(text))
    assert result.valid is expected_valid


def test_dynaguard_fails_closed() -> None:
    instance = object.__new__(DynaGuard)
    result = instance._post_processing(_gen("no verdict here"))
    assert result.valid is False
    assert result.extra == {"parse_failure": True}


def test_dynaguard_strips_explanation_and_takes_final_bare_verdict() -> None:
    instance = object.__new__(DynaGuard)
    # Reasoning inside <explanation> mentions FAIL; the actual verdict is PASS.
    explained = (
        "<explanation>This would be a FAIL if it leaked data, but it does not.</explanation>\n<answer>PASS</answer>"
    )
    assert instance._post_processing(_gen(explained)).valid is True
    # Bare fallback (no <answer>): the LAST verdict wins over a mid-reasoning mention.
    bare = "Rule 1 could be a FAIL in some readings. Overall the agent complied: PASS"
    assert instance._post_processing(_gen(bare)).valid is True


def test_wildguard_fails_closed_when_judged_response_verdict_missing() -> None:
    instance = object.__new__(WildGuard)
    # output_text was supplied (has_response) but the response-harm line didn't parse.
    data = {"generated_text": "Harmful request: no\nResponse refusal: no", "has_response": True}
    result = instance._post_processing(GuardrailInferenceOutput(data=data))
    assert result.valid is False
    assert result.extra == {"parse_failure": True}


def test_wildguard_prompt_only_tolerates_missing_response_verdict() -> None:
    instance = object.__new__(WildGuard)
    data = {"generated_text": "Harmful request: no", "has_response": False}
    assert instance._post_processing(GuardrailInferenceOutput(data=data)).valid is True


@pytest.mark.parametrize(
    ("text", "expected_valid"),
    [
        ("Prompt harm: unharmful\nResponse Harm: unharmful", True),
        ("Prompt harm: harmful\nResponse Harm: unharmful", False),
        ("<think>...</think>\nPrompt harm: unharmful\nResponse Harm: harmful", False),
    ],
)
def test_nemotron_parse(text: str, expected_valid: bool) -> None:
    instance = object.__new__(NemotronContentSafety)
    result = instance._post_processing(_gen(text))
    assert result.valid is expected_valid


def test_nemotron_fails_closed_when_judged_response_verdict_missing() -> None:
    instance = object.__new__(NemotronContentSafety)
    data = {"generated_text": "Prompt harm: unharmful", "has_response": True}
    result = instance._post_processing(GuardrailInferenceOutput(data=data))
    assert result.valid is False
    assert result.extra == {"parse_failure": True}


def test_kanana_safe_and_unsafe() -> None:
    instance = object.__new__(KananaSafeguard)
    instance.categories = KANANA_CATEGORIES["kakaocorp/kanana-safeguard-8b"]
    safe = instance._post_processing(_gen("<SAFE>"))
    unsafe = instance._post_processing(_gen("<UNSAFE-S1>"))
    assert safe.valid is True
    assert unsafe.valid is False
    assert unsafe.categories[0].name == "S1"
    assert unsafe.categories[0].description == "Hate"


def test_kanana_fails_closed() -> None:
    instance = object.__new__(KananaSafeguard)
    instance.categories = KANANA_CATEGORIES["kakaocorp/kanana-safeguard-8b"]
    result = instance._post_processing(_gen("unexpected"))
    assert result.valid is False
    assert result.extra == {"parse_failure": True}


@pytest.mark.parametrize(
    ("text", "expected_valid"),
    [
        ("Analysis... SAFE", True),
        ("This breaks rule 2. VIOLATION", False),
    ],
)
def test_gpt_oss_safeguard_parse(text: str, expected_valid: bool) -> None:
    instance = object.__new__(GptOssSafeguard)
    result = instance._post_processing(_gen(text))
    assert result.valid is expected_valid


def test_gpt_oss_safeguard_fails_closed() -> None:
    instance = object.__new__(GptOssSafeguard)
    result = instance._post_processing(_gen("no verdict word here"))
    assert result.valid is False
    assert result.extra == {"parse_failure": True}


def _qwen3(strict: bool = True) -> Qwen3Guard:
    instance = object.__new__(Qwen3Guard)
    instance.strict = strict
    return instance


def test_qwen3guard_safe_none_category_not_surfaced() -> None:
    result = _qwen3()._post_processing(_gen("Safety: Safe\nCategories: None"))
    assert result.valid is True
    assert result.score == 0.0
    assert result.extra == {"severity": "Safe"}
    assert result.categories == []


def test_qwen3guard_unsafe_single_category() -> None:
    result = _qwen3()._post_processing(_gen("Safety: Unsafe\nCategories: Violent"))
    assert result.valid is False
    assert result.score == 1.0
    assert result.extra == {"severity": "Unsafe"}
    assert [(c.name, c.triggered) for c in result.categories] == [("Violent", True)]


def test_qwen3guard_unsafe_multiple_categories() -> None:
    result = _qwen3()._post_processing(_gen("Safety: Unsafe\nCategories: Non-violent Illegal Acts, Jailbreak"))
    assert result.valid is False
    # "Non-violent Illegal Acts" must not also spuriously match "Violent".
    assert [c.name for c in result.categories] == ["Non-violent Illegal Acts", "Jailbreak"]


def test_qwen3guard_controversial_strict_vs_lenient() -> None:
    text = "Safety: Controversial\nCategories: Politically Sensitive Topics"
    strict = _qwen3(strict=True)._post_processing(_gen(text))
    lenient = _qwen3(strict=False)._post_processing(_gen(text))
    assert strict.valid is False
    assert lenient.valid is True
    for result in (strict, lenient):
        assert result.score == 0.5
        assert result.extra == {"severity": "Controversial"}


@pytest.mark.parametrize(("verdict", "expected"), [("Yes", True), ("No", False)])
def test_qwen3guard_refusal_surfaced_in_response_mode(verdict: str, expected: bool) -> None:
    data = {"generated_text": f"Safety: Safe\nCategories: None\nRefusal: {verdict}", "has_response": True}
    result = _qwen3()._post_processing(GuardrailInferenceOutput(data=data))
    refusal = next(c for c in result.categories if c.name == "refusal")
    assert refusal.triggered is expected


def test_qwen3guard_prompt_mode_omits_refusal() -> None:
    # _gen carries no has_response flag -> prompt moderation; Refusal is not surfaced even if emitted.
    result = _qwen3()._post_processing(_gen("Safety: Safe\nCategories: None\nRefusal: No"))
    assert all(c.name != "refusal" for c in result.categories)


def test_qwen3guard_missing_refusal_tolerated_in_response_mode() -> None:
    data = {"generated_text": "Safety: Unsafe\nCategories: Violent", "has_response": True}
    result = _qwen3()._post_processing(GuardrailInferenceOutput(data=data))
    assert result.valid is False
    assert result.extra == {"severity": "Unsafe"}  # not a parse failure
    refusal = next(c for c in result.categories if c.name == "refusal")
    assert refusal.triggered is None


def test_qwen3guard_categories_scoped_to_categories_line() -> None:
    text = "Safety: Safe\nCategories: None\nNote: the request mentions Violent movies and PII in passing."
    result = _qwen3()._post_processing(_gen(text))
    assert result.valid is True
    assert result.categories == []


def test_qwen3guard_strips_think_block_before_parsing() -> None:
    text = "<think>Is this a Jailbreak? Safety: Unsafe seems wrong.</think>\nSafety: Safe\nCategories: None"
    result = _qwen3()._post_processing(_gen(text))
    assert result.valid is True
    assert result.categories == []


def test_qwen3guard_fails_closed() -> None:
    result = _qwen3()._post_processing(_gen("no verdict here"))
    assert result.valid is False
    assert result.extra == {"parse_failure": True}


def _qwen3_stream(strict: bool = True) -> Qwen3GuardStream:
    instance = object.__new__(Qwen3GuardStream)
    instance.strict = strict
    return instance


def _stream_data(
    prompt: tuple[str, str] = ("Safe", "None"),
    response: list[tuple[str, str, int | None, int | None]] | None = None,
    output_text: str | None = None,
) -> GuardrailInferenceOutput[Any]:
    """Mimic Qwen3GuardStream._inference output: (risk, category, start, end) per response token."""
    return GuardrailInferenceOutput(
        data={
            "prompt_result": {"risk_level": [prompt[0]], "category": [prompt[1]]},
            "response_tokens": [
                {"result": {"risk_level": [risk], "category": [category]}, "start": start, "end": end}
                for risk, category, start, end in (response or [])
            ],
            "has_response": response is not None,
            "output_text": output_text,
        }
    )


def test_qwen3guard_stream_safe_prompt() -> None:
    result = _qwen3_stream()._post_processing(_stream_data())
    assert result.valid is True
    assert result.score == 0.0
    assert result.extra == {"severity": "Safe", "prompt_severity": "Safe"}
    assert result.categories == []
    assert result.spans is None


def test_qwen3guard_stream_unsafe_prompt_with_category() -> None:
    result = _qwen3_stream()._post_processing(_stream_data(prompt=("Unsafe", "Violent")))
    assert result.valid is False
    assert result.score == 1.0
    assert result.extra == {"severity": "Unsafe", "prompt_severity": "Unsafe"}
    assert [(c.name, c.triggered) for c in result.categories] == [("Violent", True)]
    assert result.spans is None  # prompt mode judges the prompt as one unit


def test_qwen3guard_stream_response_worst_wins_dedups_and_merges_span() -> None:
    output_text = "some unsafe text"
    result = _qwen3_stream()._post_processing(
        _stream_data(
            response=[("Safe", "None", 0, 5), ("Unsafe", "Violent", 5, 11), ("Unsafe", "Violent", 11, 16)],
            output_text=output_text,
        )
    )
    assert result.valid is False
    assert result.score == 1.0
    assert result.extra == {"severity": "Unsafe", "prompt_severity": "Safe", "response_severity": "Unsafe"}
    assert [c.name for c in result.categories] == ["Violent"]  # deduplicated across tokens
    assert result.spans is not None
    span = result.spans[0]
    assert (span.start, span.end, span.text, span.label, span.score) == (5, 16, "unsafe text", "Violent", 1.0)
    assert len(result.spans) == 1  # consecutive same-verdict tokens merge into one span


def test_qwen3guard_stream_span_splits_on_verdict_change() -> None:
    result = _qwen3_stream()._post_processing(
        _stream_data(
            response=[("Unsafe", "Violent", 0, 4), ("Controversial", "PII", 4, 8)],
            output_text="abcdefgh",
        )
    )
    assert result.spans is not None
    assert [(s.start, s.end, s.label, s.score) for s in result.spans] == [(0, 4, "Violent", 1.0), (4, 8, "PII", 0.5)]
    assert [c.name for c in result.categories] == ["Violent", "PII"]


def test_qwen3guard_stream_offsetless_tokens_flagged_but_not_spanned() -> None:
    # Scaffolding tokens (start=None) still count toward the verdict but split/skip spans.
    result = _qwen3_stream()._post_processing(
        _stream_data(
            response=[("Unsafe", "Violent", 0, 4), ("Unsafe", "Violent", None, None), ("Unsafe", "Violent", 6, 9)],
            output_text="abcdefghi",
        )
    )
    assert result.valid is False
    assert result.spans is not None
    assert [(s.start, s.end) for s in result.spans] == [(0, 4), (6, 9)]


def test_qwen3guard_stream_controversial_strict_vs_lenient() -> None:
    strict = _qwen3_stream(strict=True)._post_processing(_stream_data(prompt=("Controversial", "None")))
    lenient = _qwen3_stream(strict=False)._post_processing(_stream_data(prompt=("Controversial", "None")))
    assert strict.valid is False
    assert lenient.valid is True
    for result in (strict, lenient):
        assert result.score == 0.5


def test_qwen3guard_stream_fails_closed_on_missing_prompt_risk() -> None:
    data = _stream_data()
    data.data["prompt_result"] = {"risk_level": []}
    result = _qwen3_stream()._post_processing(data)
    assert result.valid is False
    assert result.extra == {"parse_failure": True}


def test_qwen3guard_stream_fails_closed_on_missing_response_risk() -> None:
    data = _stream_data(response=[("Safe", "None", 0, 4)])
    data.data["response_tokens"][0]["result"] = {}
    result = _qwen3_stream()._post_processing(data)
    assert result.valid is False
    assert result.extra == {"parse_failure": True}


# --- Rubric judges -------------------------------------------------------------


def _judge(cls: type[Any], **attrs: Any) -> Any:
    instance: Any = object.__new__(cls)
    instance.pass_threshold = attrs.get("pass_threshold", 3)
    instance.higher_is_better = attrs.get("higher_is_better", True)
    return instance


def test_prometheus_parse_and_score() -> None:
    judge = _judge(Prometheus, pass_threshold=3)
    passing = judge._post_processing(_gen("Feedback: solid answer [RESULT] 4"))
    failing = judge._post_processing(_gen("Feedback: weak [RESULT] 2"))
    assert passing.valid is True
    assert passing.extra is not None
    assert passing.extra["rubric_score"] == 4
    assert passing.score == pytest.approx(1.0 - (4 - 1) / 4)  # higher_is_better risk
    assert failing.valid is False


def test_prometheus_uses_final_result_not_inline_reference() -> None:
    """Feedback that references a rubric level inline must not be mistaken for the verdict."""
    judge = _judge(Prometheus, pass_threshold=3)
    text = "Feedback: a Score: 2 response would lack detail, but this one is solid. [RESULT] 4"
    result = judge._post_processing(_gen(text))
    assert result.extra is not None
    assert result.extra["rubric_score"] == 4
    assert result.valid is True


def test_prometheus_fails_closed() -> None:
    judge = _judge(Prometheus)
    result = judge._post_processing(_gen("no result marker"))
    assert result.valid is False
    assert result.extra == {"parse_failure": True}


def test_selene_parse() -> None:
    judge = _judge(Selene, pass_threshold=3)
    passing = judge._post_processing(_gen("**Reasoning:** good\n\n**Result:** 5"))
    failing = judge._post_processing(_gen("**Reasoning:** bad\n\n**Result:** 1"))
    assert passing.valid is True
    assert failing.valid is False
    assert isinstance(passing, GuardrailOutput)


def test_selene_fails_closed() -> None:
    judge = _judge(Selene)
    result = judge._post_processing(_gen("**Reasoning:** missing result"))
    assert result.valid is False
    assert result.extra == {"parse_failure": True}


def test_compass_judger_rating_parse() -> None:
    judge = _judge(CompassJudger, pass_threshold=6)
    passing = judge._post_processing(_gen("Good response. Rating: [[8]]"))
    failing = judge._post_processing(_gen("Weak. Rating: [[3]]"))
    assert passing.valid is True
    assert failing.valid is False


def test_compass_judger_uses_final_rating() -> None:
    """A bracketed number quoted mid-justification must not override the final rating."""
    judge = _judge(CompassJudger, pass_threshold=6)
    result = judge._post_processing(_gen("I considered scores [[3]] and [[8]]. Rating: [[7]]"))
    assert result.extra is not None
    assert result.extra["rubric_score"] == 7
    assert result.valid is True


def test_compass_judger_out_of_range_fails_closed() -> None:
    judge = _judge(CompassJudger, pass_threshold=6)
    result = judge._post_processing(_gen("Rating: [[99]]"))
    assert result.valid is False
    assert result.extra == {"parse_failure": True}


def test_compass_judger_no_rating_fails_closed() -> None:
    judge = _judge(CompassJudger, pass_threshold=6)
    result = judge._post_processing(_gen("I cannot decide"))
    assert result.valid is False
    assert result.extra == {"parse_failure": True}
