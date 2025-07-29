import pytest
from unittest.mock import patch, MagicMock
from any_guardrail import call_guardrail
from any_guardrail.guardrails.guardrail import Guardrail


class MockGuardrail(Guardrail):
    """Mock guardrail class for testing that doesn't require actual model loading."""

    def __init__(self, modelpath: str):
        super().__init__(modelpath)
        self.model = MagicMock()  # Mock model object

    def classify(self, input_text: str) -> str:
        """Mock classify method that returns a safe result."""
        return "SAFE"

    def _model_instantiation(self):
        """Mock model instantiation that returns a mock object."""
        return MagicMock()


def test_integration_instantiate_and_classify():
    """Test integration of guardrail instantiation and classification."""
    # Patch the instantiate_guardrail function to return our mock guardrail
    with patch("any_guardrail.instantiate_guardrail.instantiate_guardrail") as mock_instantiate:
        mock_instantiate.return_value = MockGuardrail("dummy-path")

        # List available guardrails
        classes = call_guardrail.list_guardrail_classes()
        assert len(classes) > 0

        # Test instantiation of a guardrail
        guardrail = call_guardrail.instantiate_guardrail("jasper", modelpath="dummy-path")

        # Test that the guardrail is properly instantiated
        assert isinstance(guardrail, MockGuardrail)
        assert guardrail.modelpath == "dummy-path"

        # Test classification
        result = guardrail.classify("test input")
        assert result == "SAFE"


def test_integration_error_propagation():
    """Test that errors are properly propagated from our library components."""
    # Test instantiation failure
    with patch(
        "any_guardrail.instantiate_guardrail.instantiate_guardrail", side_effect=ValueError("Guardrail not found")
    ):
        with pytest.raises(ValueError, match="Guardrail not found"):
            call_guardrail.instantiate_guardrail("nonexistent", modelpath="dummy-path")

    # Test classification failure with a mock guardrail that raises an exception
    class FailingMockGuardrail(MockGuardrail):
        def classify(self, input_text: str) -> str:
            raise RuntimeError("Classification failed")

    with patch("any_guardrail.instantiate_guardrail.instantiate_guardrail") as mock_instantiate:
        mock_instantiate.return_value = FailingMockGuardrail("dummy-path")

        guardrail = call_guardrail.instantiate_guardrail("jasper", modelpath="dummy-path")

        with pytest.raises(RuntimeError, match="Classification failed"):
            guardrail.classify("test input")


def test_integration_guardrail_interface_consistency():
    """Test that all guardrails have consistent interfaces."""
    classes = call_guardrail.list_guardrail_classes()
    assert len(classes) > 0

    with patch("any_guardrail.instantiate_guardrail.instantiate_guardrail") as mock_instantiate:
        mock_instantiate.return_value = MockGuardrail("dummy-path")

        # Test with first available guardrail
        guardrail_name = classes[0]
        guardrail = call_guardrail.instantiate_guardrail(guardrail_name, modelpath="dummy-path")

        # Test that all guardrails have required attributes and methods
        assert hasattr(guardrail, "modelpath")
        assert hasattr(guardrail, "model")
        assert hasattr(guardrail, "classify")
        assert callable(getattr(guardrail, "classify"))

        # Test that classify method can be called
        result = guardrail.classify("test input")
        assert result is not None
