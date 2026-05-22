"""Unit tests for the CleanlabTlm guardrail.

These tests mock the underlying ``cleanlab_tlm.TLM`` client so no real API calls are made.
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any
from unittest import mock

import pytest

from any_guardrail.base import GuardrailOutput


def _install_fake_cleanlab_tlm(monkeypatch: pytest.MonkeyPatch) -> mock.MagicMock:
    """Install a fake ``cleanlab_tlm`` module exposing a mocked ``TLM`` class.

    Returns the ``TLM`` class mock so individual tests can configure its instances.
    """
    fake_module = ModuleType("cleanlab_tlm")
    tlm_cls = mock.MagicMock(name="TLM")
    fake_module.TLM = tlm_cls  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cleanlab_tlm", fake_module)
    return tlm_cls


def _make_guardrail(tlm_cls: mock.MagicMock, score_payload: dict[str, Any]) -> Any:
    """Construct a CleanlabTlm guardrail whose underlying client returns ``score_payload``."""
    from any_guardrail.guardrails.cleanlab_tlm.cleanlab_tlm import CleanlabTlm

    tlm_instance = mock.MagicMock()
    tlm_instance.get_trustworthiness_score.return_value = score_payload
    tlm_cls.return_value = tlm_instance
    return CleanlabTlm(api_key="fake-key")


def test_high_trust_score_above_threshold_is_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    """A trustworthiness score above the default threshold should yield ``valid=True``."""
    tlm_cls = _install_fake_cleanlab_tlm(monkeypatch)
    guardrail = _make_guardrail(
        tlm_cls,
        {"trustworthiness_score": 0.9, "log": {"explanation": "Looks solid."}},
    )

    result = guardrail.validate(prompt="What is 2+2?", response="4")

    assert isinstance(result, GuardrailOutput)
    assert result.valid is True
    assert result.score == pytest.approx(0.9)
    assert result.explanation is not None
    assert result.explanation["trustworthiness_score"] == pytest.approx(0.9)
    assert result.explanation["explanation"] == "Looks solid."


def test_low_trust_score_below_threshold_is_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    """A trustworthiness score below the default threshold should yield ``valid=False``."""
    tlm_cls = _install_fake_cleanlab_tlm(monkeypatch)
    guardrail = _make_guardrail(
        tlm_cls,
        {"trustworthiness_score": 0.4, "log": {"explanation": "Uncertain."}},
    )

    result = guardrail.validate(prompt="What is 2+2?", response="5")

    assert isinstance(result, GuardrailOutput)
    assert result.valid is False
    assert result.score == pytest.approx(0.4)
    assert result.explanation is not None
    assert result.explanation["trustworthiness_score"] == pytest.approx(0.4)


def test_custom_threshold_is_respected(monkeypatch: pytest.MonkeyPatch) -> None:
    """A score below an explicit, stricter threshold should yield ``valid=False``."""
    tlm_cls = _install_fake_cleanlab_tlm(monkeypatch)
    guardrail = _make_guardrail(
        tlm_cls,
        {"trustworthiness_score": 0.9, "log": {}},
    )

    result = guardrail.validate(prompt="Q", response="A", threshold=0.95)

    assert isinstance(result, GuardrailOutput)
    assert result.valid is False
    assert result.score == pytest.approx(0.9)


def test_api_key_from_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """Constructor should fall back to ``CLEANLAB_TLM_API_KEY`` when ``api_key`` is omitted."""
    tlm_cls = _install_fake_cleanlab_tlm(monkeypatch)
    tlm_cls.return_value = mock.MagicMock()
    monkeypatch.setenv("CLEANLAB_TLM_API_KEY", "env-key-123")

    from any_guardrail.guardrails.cleanlab_tlm.cleanlab_tlm import CleanlabTlm

    guardrail = CleanlabTlm()

    assert guardrail.api_key == "env-key-123"
    # TLM client should have been instantiated with the env-var key.
    _, kwargs = tlm_cls.call_args
    assert kwargs["api_key"] == "env-key-123"


def test_missing_api_key_raises_informative_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Constructor should raise ``ValueError`` mentioning the env var when no key is available."""
    _install_fake_cleanlab_tlm(monkeypatch)
    monkeypatch.delenv("CLEANLAB_TLM_API_KEY", raising=False)

    from any_guardrail.guardrails.cleanlab_tlm.cleanlab_tlm import CleanlabTlm

    with pytest.raises(ValueError, match="CLEANLAB_TLM_API_KEY"):
        CleanlabTlm()


def test_invalid_model_id_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Passing an unknown ``model_id`` should raise ``ValueError``."""
    _install_fake_cleanlab_tlm(monkeypatch)

    from any_guardrail.guardrails.cleanlab_tlm.cleanlab_tlm import CleanlabTlm

    with pytest.raises(ValueError, match="Only supports"):
        CleanlabTlm(model_id="not-a-real-model", api_key="fake")


def test_get_trustworthiness_score_called_with_prompt_and_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """``validate`` should forward its arguments to the underlying TLM client."""
    tlm_cls = _install_fake_cleanlab_tlm(monkeypatch)
    guardrail = _make_guardrail(
        tlm_cls,
        {"trustworthiness_score": 0.8, "log": {}},
    )

    guardrail.validate(prompt="my prompt", response="my response")

    guardrail.tlm.get_trustworthiness_score.assert_called_once_with(
        prompt="my prompt",
        response="my response",
    )
