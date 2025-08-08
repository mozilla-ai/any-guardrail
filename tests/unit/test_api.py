from pathlib import Path
from any_guardrail.guardrail import GuardrailName


def test_all_guardrails_in_enum() -> None:
    """Test that all guardrail modules are accounted for in the GuardrailName enum."""
    guardrails_dir = Path(__file__).parent.parent.parent / "src" / "any_guardrail" / "guardrails"

    # Take all .py modules except dunder and cache
    guardrail_modules = []
    for item in guardrails_dir.iterdir():
        if item.is_file() and item.suffix == ".py" and item.stem != "__init__":
            guardrail_modules.append(item.stem)

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
        if item.is_file() and item.suffix == ".py" and item.stem != "__init__":
            actual_modules.add(item.stem)

    enum_modules = {provider.value for provider in GuardrailName}

    assert actual_modules == enum_modules, (
        "Guardrail modules and enum values don't match!\n"
        f"In modules but not enum: {actual_modules - enum_modules}\n"
        f"In enum but not modules: {enum_modules - actual_modules}"
    )
