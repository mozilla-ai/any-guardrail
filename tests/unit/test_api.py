from pathlib import Path
from unittest.mock import patch

import pytest

from any_guardrail import AnyGuardrail, GuardrailName
from any_guardrail.base import Guardrail
from any_guardrail.guardrails.any_llm import AnyLlm
from any_guardrail.guardrails.huggingface import HuggingFace


def test_all_guardrails_in_enum() -> None:
    """Test that all guardrail modules are accounted for in the GuardrailName enum."""
    guardrails_dir = Path(__file__).parent.parent.parent / "src" / "any_guardrail" / "guardrails"

    guardrail_modules = []
    for item in guardrails_dir.iterdir():
        if item.is_dir() and item.name not in ("__pycache__", "huggingface"):
            for sub_item in item.iterdir():
                if (
                    sub_item.is_file()
                    and sub_item.suffix == ".py"
                    and sub_item.stem not in ("__init__", "off_topic_stsb", "off_topic_jina")
                ):
                    guardrail_modules.append(sub_item.stem)

    enum_values = [provider.value for provider in GuardrailName]

    guardrail_modules.sort()
    enum_values.sort()

    missing_from_enum = set(guardrail_modules) - set(enum_values)
    missing_from_modules = set(enum_values) - set(guardrail_modules)

    assert not missing_from_enum, f"Guardrail modules missing from GuardrailName enum: {missing_from_enum}"
    assert not missing_from_modules, f"GuardrailName enum values missing guardrail modules: {missing_from_modules}"

    assert guardrail_modules == enum_values, (
        f"Guardrail modules {guardrail_modules} don't match enum values {enum_values}"
    )


def test_guardrail_enum_values_match_module_names() -> None:
    """Test that enum values exactly match guardrail module file names (without .py)."""
    guardrails_dir = Path(__file__).parent.parent.parent / "src" / "any_guardrail" / "guardrails"

    actual_modules = set()
    for item in guardrails_dir.iterdir():
        if item.is_dir() and item.name not in ("__pycache__", "huggingface"):
            for sub_item in item.iterdir():
                if (
                    sub_item.is_file()
                    and sub_item.suffix == ".py"
                    and sub_item.stem not in ("__init__", "off_topic_stsb", "off_topic_jina")
                ):
                    actual_modules.add(sub_item.stem)

    enum_modules = {provider.value for provider in GuardrailName}

    assert actual_modules == enum_modules, (
        "Guardrail modules and enum values don't match!\n"
        f"In modules but not enum: {actual_modules - enum_modules}\n"
        f"In enum but not modules: {enum_modules - actual_modules}"
    )


def test_create_guardrail_with_invalid_id_raises_error() -> None:
    """Test that creating a guardrail with an invalid model ID raises a ValueError."""
    with pytest.raises(ValueError, match="Only supports"):
        AnyGuardrail.create(guardrail_name=GuardrailName.SHIELD_GEMMA, model_id="invalid_id", policy="Help")


def test_get_guardrail_class_all_valid_names() -> None:
    """Test that all guardrail names can be resolved to their classes."""
    for guardrail_name in GuardrailName:
        guardrail_class = AnyGuardrail._get_guardrail_class(guardrail_name)
        assert guardrail_class is not None, f"Guardrail class for {guardrail_name} could not be resolved"
        assert issubclass(guardrail_class, Guardrail), (
            f"Guardrail class {guardrail_class} for {guardrail_name} is not a subclass of Guardrail"
        )


def test_model_load() -> None:
    """Test that all guardrail models load the backend model on instantiation."""
    for guardrail_name in GuardrailName:
        guardrail_class = AnyGuardrail._get_guardrail_class(guardrail_name)
        if guardrail_class is not AnyLlm:
            with patch.object(guardrail_class, "_load_model") as mock_load_model:
                if guardrail_name == GuardrailName.FLOWJUDGE:
                    mock_load_model.return_value = "mocked_model"
                    name = "Dummy"
                    criteria = "Dummy criteria"
                    rubric = {0: "Dummy 0", 1: "Dummy 1"}
                    required_inputs = ["dummy"]
                    required_output = "response"
                    guardrail = AnyGuardrail.create(
                        guardrail_name=guardrail_name,
                        name=name,
                        criteria=criteria,
                        rubric=rubric,
                        required_inputs=required_inputs,
                        required_output=required_output,
                    )
                else:
                    mock_load_model.side_effect = lambda: setattr(guardrail_class, "model", "mocked_model")  # noqa: B023
                    if guardrail_name == GuardrailName.SHIELD_GEMMA:
                        guardrail = AnyGuardrail.create(guardrail_name=guardrail_name, policy="Dummy")
                    elif guardrail_name == GuardrailName.GLIDER:
                        guardrail = AnyGuardrail.create(
                            guardrail_name=guardrail_name,
                            pass_criteria="Dummy",  # noqa: S106
                            rubric="Dummy",
                        )
                    else:
                        guardrail = AnyGuardrail.create(guardrail_name=guardrail_name)
                assert guardrail.model == "mocked_model"  # type: ignore[attr-defined]


def test_post_processing_implementation() -> None:
    """Test that all guardrail models that inherit from HuggingFace implement the _post_processing method."""
    for guardrail_name in GuardrailName:
        guardrail_class = AnyGuardrail._get_guardrail_class(guardrail_name)
        if issubclass(guardrail_class, HuggingFace):
            assert "_post_processing" in guardrail_class.__dict__, (
                f"Guardrail class {guardrail_class} for {guardrail_name} does not have a _post_processing method"
            )
