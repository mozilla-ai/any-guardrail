import pytest
from abc import ABC
from any_guardrail.guardrails.guardrail import Guardrail
from any_guardrail.utils.custom_types import ClassificationOutput, GuardrailModel


class ConcreteGuardrail(Guardrail):
    """Concrete implementation of Guardrail for testing."""
    
    def __init__(self, model_identifier: str):
        super().__init__(model_identifier)
    
    def safety_review(self) -> ClassificationOutput:
        """Concrete implementation of safety_review."""
        return ClassificationOutput(unsafe=False, explanation="Test", score=0.0)
    
    def _model_instantiation(self) -> GuardrailModel:
        """Concrete implementation of _model_instantiation."""
        return GuardrailModel()


class TestGuardrailBase:
    """Test cases for the base Guardrail class."""

    def test_guardrail_is_abstract(self):
        """Test that Guardrail is an abstract base class."""
        assert issubclass(Guardrail, ABC)

    def test_guardrail_has_abstract_methods(self):
        """Test that Guardrail has the required abstract methods."""
        assert hasattr(Guardrail, 'safety_review')
        assert hasattr(Guardrail, '_model_instantiation')
        
        # Check that these are abstract methods
        assert Guardrail.safety_review.__isabstractmethod__
        assert Guardrail._model_instantiation.__isabstractmethod__

    def test_concrete_guardrail_instantiation(self):
        """Test that a concrete implementation can be instantiated."""
        guardrail = ConcreteGuardrail("test/model")
        assert guardrail.model_identifier == "test/model"

    def test_concrete_guardrail_methods(self):
        """Test that concrete implementation methods work correctly."""
        guardrail = ConcreteGuardrail("test/model")
        
        # Test safety_review
        result = guardrail.safety_review()
        assert isinstance(result, ClassificationOutput)
        assert result.unsafe is False
        assert result.explanation == "Test"
        assert result.score == 0.0
        
        # Test _model_instantiation
        model_result = guardrail._model_instantiation()
        assert isinstance(model_result, GuardrailModel)

    def test_guardrail_init_with_different_identifiers(self):
        """Test that Guardrail can be initialized with different model identifiers."""
        identifiers = [
            "test/model",
            "deepset/deberta-v3-base-injection",
            "DuoGuard/DuoGuard-0.5B",
            "",
            "model_with_underscores",
            "model-with-dashes",
            "model.with.dots"
        ]
        
        for identifier in identifiers:
            guardrail = ConcreteGuardrail(identifier)
            assert guardrail.model_identifier == identifier

    def test_guardrail_model_identifier_attribute(self):
        """Test that model_identifier is properly set as an attribute."""
        guardrail = ConcreteGuardrail("test/model")
        assert hasattr(guardrail, 'model_identifier')
        assert guardrail.model_identifier == "test/model"

    def test_guardrail_inheritance_structure(self):
        """Test that Guardrail has the correct inheritance structure."""
        # Test that ConcreteGuardrail inherits from Guardrail
        assert issubclass(ConcreteGuardrail, Guardrail)
        
        # Test that Guardrail is abstract
        with pytest.raises(TypeError):
            Guardrail("test/model")