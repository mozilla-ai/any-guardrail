import pytest
from unittest.mock import Mock, patch
from any_guardrail.call_guardrail import CallGuardrail
from any_guardrail.guardrails.guardrail import Guardrail


class TestCallGuardrail:
    """Test cases for the CallGuardrail class."""

    def test_init(self):
        """Test that CallGuardrail initializes correctly."""
        call_guardrail = CallGuardrail()
        assert hasattr(call_guardrail, 'model_registry')
        assert call_guardrail.model_registry is not None

    def test_list_all_supported_guardrails(self):
        """Test that list_all_supported_guardrails returns the correct list."""
        call_guardrail = CallGuardrail()
        supported_guardrails = call_guardrail.list_all_supported_guardrails()
        
        # Check that it returns a list
        assert isinstance(supported_guardrails, list)
        
        # Check that it contains expected guardrails
        expected_guardrails = [
            "deepset/deberta-v3-base-injection",
            "DuoGuard/DuoGuard-0.5B",
            "DuoGuard/DuoGuard-1B-Llama-3.2-transfer",
            "DuoGuard/DuoGuard-1.5B-transfer",
            "Flowjudge",
            "flowjudge",
            "FlowJudge",
            "PatronusAI/glider",
            "hbseong/HarmAug-Guard",
            "leolee99/InjecGuard",
            "JasperLS/deberta-v3-base-injection",
            "JasperLS/gelectra-base-injection",
            "dcarpintero/pangolin-guard-large",
            "dcarpintero/pangolin-guard-base",
            "protectai/deberta-v3-base-prompt-injection",
            "protectai/deberta-v3-small-prompt-injection-v2",
            "protectai/deberta-v3-base-prompt-injection-v2",
            "qualifire/prompt-injection-sentinel",
            "google/shieldgemma-2b",
            "google/shieldgemma-9b",
            "google/shieldgemma-27b",
        ]
        
        for expected in expected_guardrails:
            assert expected in supported_guardrails

    def test_call_guardrail_invalid_identifier(self):
        """Test that call_guardrail raises ValueError for invalid identifiers."""
        call_guardrail = CallGuardrail()
        
        with pytest.raises(ValueError) as exc_info:
            call_guardrail.call_guardrail("invalid/guardrail")
        
        assert "We do not support the model you are trying to instantiate" in str(exc_info.value)
        assert "list_all_support_guardrails" in str(exc_info.value)

    def test_call_guardrail_empty_identifier(self):
        """Test that call_guardrail raises ValueError for empty identifier."""
        call_guardrail = CallGuardrail()
        
        with pytest.raises(ValueError) as exc_info:
            call_guardrail.call_guardrail("")
        
        assert "We do not support the model you are trying to instantiate" in str(exc_info.value)

    def test_call_guardrail_none_identifier(self):
        """Test that call_guardrail raises ValueError for None identifier."""
        call_guardrail = CallGuardrail()
        
        with pytest.raises(ValueError) as exc_info:
            call_guardrail.call_guardrail(None)  # type: ignore
        
        assert "We do not support the model you are trying to instantiate" in str(exc_info.value)
         