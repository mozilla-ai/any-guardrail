import pytest
from unittest.mock import Mock, patch
from any_guardrail.call_guardrail import CallGuardrail
from any_guardrail.guardrails.guardrail import Guardrail
from any_guardrail.utils.custom_types import ClassificationOutput, GuardrailModel


class TestErrorHandling:
    """Test cases for error handling and edge cases."""

    def test_call_guardrail_with_none_registry(self):
        """Test CallGuardrail behavior when model_registry is None."""
        call_guardrail = CallGuardrail()
        call_guardrail.model_registry = None
        
        with pytest.raises(AttributeError):
            call_guardrail.list_all_supported_guardrails()

    def test_call_guardrail_with_empty_registry(self):
        """Test CallGuardrail behavior when model_registry is empty."""
        call_guardrail = CallGuardrail()
        call_guardrail.model_registry = {}
        
        supported_guardrails = call_guardrail.list_all_supported_guardrails()
        assert supported_guardrails == []
        
        with pytest.raises(ValueError) as exc_info:
            call_guardrail.call_guardrail("any/model")
        
        assert "We do not support the model you are trying to instantiate" in str(exc_info.value)

    def test_call_guardrail_with_non_dict_registry(self):
        """Test CallGuardrail behavior when model_registry is not a dict."""
        call_guardrail = CallGuardrail()
        call_guardrail.model_registry = "not_a_dict"
        
        with pytest.raises(AttributeError):
            call_guardrail.list_all_supported_guardrails()

    def test_guardrail_instantiation_with_invalid_class(self):
        """Test behavior when registry contains invalid class."""
        call_guardrail = CallGuardrail()
        call_guardrail.model_registry = {"test/model": str}  # str is not a guardrail class
        
        with pytest.raises(TypeError):
            call_guardrail.call_guardrail("test/model")

    def test_guardrail_with_missing_required_methods(self):
        """Test behavior when guardrail class is missing required methods."""
        class IncompleteGuardrail(Guardrail):
            def __init__(self, model_identifier: str):
                super().__init__(model_identifier)
            
            # Missing safety_review and _model_instantiation methods
        
        # This should raise TypeError when trying to instantiate
        with pytest.raises(TypeError):
            IncompleteGuardrail("test/model")

    def test_classification_output_equality_with_different_types(self):
        """Test ClassificationOutput equality with different types."""
        output1 = ClassificationOutput(unsafe=True, explanation="test", score=0.5)
        output2 = ClassificationOutput(unsafe=True, explanation="test", score=0.5)
        output3 = ClassificationOutput(unsafe=False, explanation="test", score=0.5)
        
        assert output1 == output2
        assert output1 != output3
        assert output1 != "not_a_classification_output"

    def test_guardrail_model_equality_with_different_types(self):
        """Test GuardrailModel equality with different types."""
        model1 = GuardrailModel(model="test", tokenizer="test")
        model2 = GuardrailModel(model="test", tokenizer="test")
        model3 = GuardrailModel(model="different", tokenizer="test")
        
        assert model1 == model2
        assert model1 != model3
        assert model1 != "not_a_guardrail_model"
        