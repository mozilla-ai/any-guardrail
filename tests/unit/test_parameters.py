"""Parity and invariant tests for the machine-readable parameter schema (issue #206).

These guarantee the generated parameter data cannot silently drift from the guardrail
signatures, that the typed specs stay consistent with the taxonomy's validate-kwargs and
each guardrail's ``SUPPORTED_MODELS``, and that the registry stays import-free.
"""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

import any_guardrail._parameter_data as parameter_data_module
import any_guardrail.parameters as parameters_module
from any_guardrail import AnyGuardrail, GuardrailName
from any_guardrail.parameter_registry import PARAMETER_REGISTRY, get_parameter_schema
from any_guardrail.parameters import ParameterSpec, ParameterStage, ParameterType
from any_guardrail.registry import GUARDRAIL_METADATA

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

import generate_parameter_data
import generate_parameters_json

ALL_NAMES = list(GuardrailName)


def test_registry_covers_all_guardrails_exactly() -> None:
    """Every GuardrailName has exactly one parameter-registry entry, and vice versa."""
    assert set(PARAMETER_REGISTRY) == set(GuardrailName)


@pytest.mark.parametrize("name", ALL_NAMES, ids=lambda n: n.value)
def test_get_parameter_schema_returns_specs(name: GuardrailName) -> None:
    """The accessor returns a list of ParameterSpec (possibly empty)."""
    specs = get_parameter_schema(name)
    assert isinstance(specs, list)
    assert all(isinstance(spec, ParameterSpec) for spec in specs)


def test_parameter_data_matches_fresh_generation() -> None:
    """The committed _parameter_data.py is exactly what the generator produces (no drift)."""
    expected = generate_parameter_data.render(generate_parameter_data.build_payload())
    committed = Path(parameter_data_module.__file__).read_text(encoding="utf-8")
    assert committed == expected, "run `python scripts/generate_parameter_data.py` and commit the result"


def test_parameters_json_matches_registry() -> None:
    """The committed schemas/guardrail_parameters.json matches the registry export."""
    expected = generate_parameters_json.render(generate_parameters_json.build_payload())
    committed = generate_parameters_json.DEFAULT_OUT.read_text(encoding="utf-8")
    assert committed == expected, "run `python scripts/generate_parameters_json.py` and commit the result"


@pytest.mark.parametrize("name", ALL_NAMES, ids=lambda n: n.value)
def test_validate_specs_reconcile_with_metadata(name: GuardrailName) -> None:
    """Validate-stage specs partition the taxonomy's recorded validate kwargs."""
    validate_specs = [spec for spec in get_parameter_schema(name) if spec.stage is ParameterStage.VALIDATE]
    meta = GUARDRAIL_METADATA[name]
    spec_names = {spec.name for spec in validate_specs}
    assert spec_names == meta.required_validate_kwargs | meta.optional_validate_kwargs
    # A parameter the signature forces (no default) is recorded as required in the taxonomy too.
    signature_required = {spec.name for spec in validate_specs if spec.required}
    assert signature_required <= meta.required_validate_kwargs


@pytest.mark.parametrize("name", ALL_NAMES, ids=lambda n: n.value)
def test_model_id_enum_choices_match_supported_models(name: GuardrailName) -> None:
    """A create-stage model_id enum lists exactly the guardrail's SUPPORTED_MODELS."""
    model_id_specs = [
        spec
        for spec in get_parameter_schema(name)
        if spec.name == "model_id" and spec.stage is ParameterStage.CREATE and spec.type is ParameterType.ENUM
    ]
    if not model_id_specs:
        return
    supported = tuple(AnyGuardrail._get_guardrail_class(name).SUPPORTED_MODELS)
    assert model_id_specs[0].choices == supported


@pytest.mark.parametrize("name", ALL_NAMES, ids=lambda n: n.value)
def test_enum_specs_have_choices_and_others_do_not(name: GuardrailName) -> None:
    """Every enum spec carries non-empty choices; every non-enum spec carries none."""
    for spec in get_parameter_schema(name):
        if spec.type is ParameterType.ENUM:
            assert spec.choices, f"{name.value}:{spec.name} is enum without choices"
        else:
            assert spec.choices is None, f"{name.value}:{spec.name} is {spec.type} but has choices"


def _import_roots(module_file: str) -> set[str]:
    tree = ast.parse(Path(module_file).read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_parameters_module_is_leaf() -> None:
    """parameters.py depends only on the stdlib and pydantic."""
    roots = _import_roots(parameters_module.__file__)
    assert roots <= {"enum", "typing", "pydantic"}, f"parameters.py imports beyond stdlib/pydantic: {roots}"


def test_parameter_data_module_is_stdlib_leaf() -> None:
    """The generated _parameter_data.py depends only on the stdlib."""
    roots = _import_roots(parameter_data_module.__file__)
    assert roots <= {"json", "typing"}, f"_parameter_data.py imports beyond stdlib: {roots}"


def test_get_parameter_schema_loads_no_guardrail_modules() -> None:
    """Reading the parameter schema never imports a guardrail implementation module."""
    code = (
        "import sys\n"
        "from any_guardrail import AnyGuardrail, GuardrailName\n"
        "for n in GuardrailName:\n"
        "    AnyGuardrail.get_parameter_schema(n)\n"
        "impl = [m for m in sys.modules if m.startswith('any_guardrail.guardrails.')]\n"
        "assert impl == [], impl\n"
        "print('ok')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)  # noqa: S603
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout
