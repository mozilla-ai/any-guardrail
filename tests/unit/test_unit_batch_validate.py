"""Batch-dispatch tests for the generative judges with real ``_validate_batch`` overrides (issue #227).

Covers Prometheus, DynaGuard, and CompassJudger (GraniteGuardian has its own dedicated
batch tests in ``test_unit_granite_guardian.py``). These exercise the batching contract
itself — one real ``generate_chat`` call for a ``HuggingFaceProvider``, a sequential
fallback otherwise — not the prompt-parsing logic, which is covered elsewhere.
"""

from typing import Any
from unittest.mock import MagicMock

from any_guardrail.base import GuardrailName
from any_guardrail.guardrails.compass_judger.compass_judger import CompassJudger
from any_guardrail.guardrails.dyna_guard.dyna_guard import DynaGuard
from any_guardrail.guardrails.prometheus.prometheus import Prometheus
from any_guardrail.prompt_registry import resolve_prompt
from any_guardrail.providers.huggingface import HuggingFaceProvider
from any_guardrail.types import GuardrailInferenceOutput


def _batch_output(generated_texts: list[str]) -> GuardrailInferenceOutput[Any]:
    """Mimic a batched provider.generate_chat output."""
    return GuardrailInferenceOutput(
        data={
            "generated_text": generated_texts,
            "prompt_token_count": [10] * len(generated_texts),
            "completion_token_count": [5] * len(generated_texts),
            "raw": None,
        }
    )


def _prometheus_instance() -> Any:
    instance: Any = object.__new__(Prometheus)
    instance.pass_threshold = 3
    instance.higher_is_better = True
    instance.rubric = "Score 1: bad. Score 5: good."
    instance.reference_answer = None
    instance._prompt = resolve_prompt(GuardrailName.PROMETHEUS, None, None)
    return instance


def _dyna_guard_instance() -> Any:
    instance: Any = object.__new__(DynaGuard)
    instance.think = False
    instance.policy = "1. Do not issue refunds."
    instance._prompt = resolve_prompt(GuardrailName.DYNA_GUARD, None, None)
    return instance


def _compass_judger_instance() -> Any:
    instance: Any = object.__new__(CompassJudger)
    instance.pass_threshold = 6
    instance.higher_is_better = True
    instance.criteria = "Helpfulness"
    instance.rubric = "1-3: unhelpful. 8-10: fully helpful."
    instance._prompt = resolve_prompt(GuardrailName.COMPASS_JUDGER, None, None)
    return instance


def test_prometheus_validate_batch_uses_real_batching_with_hf_provider() -> None:
    judge = _prometheus_instance()
    provider = MagicMock(spec=HuggingFaceProvider)
    provider.generate_chat.return_value = _batch_output(["Feedback: good [RESULT] 4", "Feedback: bad [RESULT] 1"])
    judge.provider = provider

    results = judge.validate(["instruction 1", "instruction 2"])

    provider.generate_chat.assert_called_once()
    assert isinstance(results, list)
    assert len(results) == 2
    assert results[0].valid is True
    assert results[1].valid is False


def test_prometheus_validate_batch_falls_back_for_non_hf_provider() -> None:
    judge = _prometheus_instance()
    provider = MagicMock()
    provider.generate_chat.return_value = GuardrailInferenceOutput(
        data={"generated_text": "Feedback: ok [RESULT] 4", "prompt_token_count": 10, "completion_token_count": 5}
    )
    judge.provider = provider

    results = judge.validate(["instruction 1", "instruction 2", "instruction 3"])

    assert provider.generate_chat.call_count == 3
    assert isinstance(results, list)
    assert len(results) == 3


def test_prometheus_validate_batch_empty_list() -> None:
    judge = _prometheus_instance()
    provider = MagicMock(spec=HuggingFaceProvider)
    judge.provider = provider

    assert judge.validate([]) == []
    provider.generate_chat.assert_not_called()


def test_dyna_guard_validate_batch_uses_real_batching_with_hf_provider() -> None:
    judge = _dyna_guard_instance()
    provider = MagicMock(spec=HuggingFaceProvider)
    provider.generate_chat.return_value = _batch_output(["<answer>PASS</answer>", "<answer>FAIL</answer>"])
    judge.provider = provider

    results = judge.validate(["transcript 1", "transcript 2"])

    provider.generate_chat.assert_called_once()
    assert isinstance(results, list)
    assert results[0].valid is True
    assert results[1].valid is False


def test_dyna_guard_validate_batch_broadcasts_single_output_text() -> None:
    judge = _dyna_guard_instance()
    provider = MagicMock(spec=HuggingFaceProvider)
    provider.generate_chat.return_value = _batch_output(["<answer>PASS</answer>", "<answer>PASS</answer>"])
    judge.provider = provider

    judge.validate(["user turn 1", "user turn 2"], output_text="shared agent reply")

    _, kwargs = provider.generate_chat.call_args
    messages_batch = kwargs["messages"]
    assert len(messages_batch) == 2
    for messages in messages_batch:
        contents = [m["content"] for m in messages]
        assert any("shared agent reply" in c for c in contents)


def test_dyna_guard_validate_batch_rejects_mismatched_output_text_length() -> None:
    judge = _dyna_guard_instance()
    provider = MagicMock(spec=HuggingFaceProvider)
    judge.provider = provider

    try:
        judge.validate(["t1", "t2"], output_text=["only one"])
    except ValueError as e:
        assert "output_text" in str(e)
    else:
        raise AssertionError("expected ValueError for mismatched output_text length")


def test_compass_judger_validate_batch_uses_real_batching_with_hf_provider() -> None:
    judge = _compass_judger_instance()
    provider = MagicMock(spec=HuggingFaceProvider)
    provider.generate_chat.return_value = _batch_output(["Good. Rating: [[8]]", "Weak. Rating: [[2]]"])
    judge.provider = provider

    results = judge.validate(["instruction 1", "instruction 2"])

    provider.generate_chat.assert_called_once()
    assert isinstance(results, list)
    assert results[0].valid is True
    assert results[1].valid is False


def test_compass_judger_validate_batch_falls_back_for_non_hf_provider() -> None:
    judge = _compass_judger_instance()
    provider = MagicMock()
    provider.generate_chat.return_value = GuardrailInferenceOutput(
        data={"generated_text": "Rating: [[8]]", "prompt_token_count": 10, "completion_token_count": 5}
    )
    judge.provider = provider

    results = judge.validate(["instruction 1", "instruction 2"])

    assert provider.generate_chat.call_count == 2
    assert isinstance(results, list)
    assert len(results) == 2
