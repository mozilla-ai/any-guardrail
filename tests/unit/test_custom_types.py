import pytest
from any_guardrail.utils.custom_types import ClassificationOutput, GuardrailModel


class TestClassificationOutput:
    """Test cases for the ClassificationOutput dataclass."""

    def test_classification_output_default_values(self):
        """Test that ClassificationOutput has correct default values."""
        output = ClassificationOutput()
        assert output.unsafe is None
        assert output.explanation is None
        assert output.score is None

    def test_classification_output_with_values(self):
        """Test that ClassificationOutput can be created with values."""
        output = ClassificationOutput(
            unsafe=True,
            explanation="Test explanation",
            score=0.85
        )
        assert output.unsafe is True
        assert output.explanation == "Test explanation"
        assert output.score == 0.85

    def test_classification_output_with_dict_explanation(self):
        """Test that ClassificationOutput can handle dict explanations."""
        explanation_dict = {"category1": True, "category2": False}
        output = ClassificationOutput(
            unsafe=True,
            explanation=explanation_dict,
            score=1
        )
        assert output.unsafe is True
        assert output.explanation == explanation_dict
        assert output.score == 1

    def test_classification_output_equality(self):
        """Test that ClassificationOutput instances can be compared."""
        output1 = ClassificationOutput(unsafe=True, explanation="Test", score=0.8)
        output2 = ClassificationOutput(unsafe=True, explanation="Test", score=0.8)
        output3 = ClassificationOutput(unsafe=False, explanation="Test", score=0.8)
        
        assert output1 == output2
        assert output1 != output3


class TestGuardrailModel:
    """Test cases for the GuardrailModel dataclass."""

    def test_guardrail_model_default_values(self):
        """Test that GuardrailModel has correct default values."""
        model = GuardrailModel()
        assert model.model is None
        assert model.tokenizer is None

    def test_guardrail_model_with_values(self):
        """Test that GuardrailModel can be created with values."""
        mock_model = object()
        mock_tokenizer = object()
        
        model = GuardrailModel(
            model=mock_model,
            tokenizer=mock_tokenizer
        )
        assert model.model is mock_model
        assert model.tokenizer is mock_tokenizer

    def test_guardrail_model_with_only_model(self):
        """Test that GuardrailModel can be created with only model."""
        mock_model = object()
        
        model = GuardrailModel(model=mock_model)
        assert model.model is mock_model
        assert model.tokenizer is None

    def test_guardrail_model_equality(self):
        """Test that GuardrailModel instances can be compared."""
        mock_model1 = object()
        mock_tokenizer1 = object()
        mock_model2 = object()
        mock_tokenizer2 = object()
        
        model1 = GuardrailModel(model=mock_model1, tokenizer=mock_tokenizer1)
        model2 = GuardrailModel(model=mock_model1, tokenizer=mock_tokenizer1)
        model3 = GuardrailModel(model=mock_model2, tokenizer=mock_tokenizer2)
        
        assert model1 == model2
        assert model1 != model3

    def test_guardrail_model_attributes_are_mutable(self):
        """Test that GuardrailModel attributes can be modified."""
        model = GuardrailModel()
        mock_model = object()
        mock_tokenizer = object()
        
        model.model = mock_model
        model.tokenizer = mock_tokenizer
        
        assert model.model is mock_model
        assert model.tokenizer is mock_tokenizer 
        