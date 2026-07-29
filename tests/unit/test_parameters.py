"""Parity and invariant tests for the machine-readable parameter schema (issue #206).

These guarantee the generated parameter data cannot silently drift from the guardrail
signatures, that the typed specs stay consistent with the taxonomy's validate-kwargs and
each guardrail's ``SUPPORTED_MODELS``, and that the registry stays import-free.
"""

import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest

import any_guardrail._parameter_data as parameter_data_module
import any_guardrail.parameters as parameters_module
from any_guardrail import AnyGuardrail, GuardrailName
from any_guardrail.parameter_registry import (
    PARAMETER_REGISTRY,
    get_parameter_schema,
    get_requirement_groups,
)
from any_guardrail.parameters import ParameterSpec, ParameterStage, ParameterType, RequirementGroup
from any_guardrail.registry import GUARDRAIL_METADATA

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

import generate_parameter_data
import generate_parameters_json

ALL_NAMES = list(GuardrailName)

_GUARDRAILS_DIR = Path(__file__).parent.parent.parent / "src" / "any_guardrail" / "guardrails"
# Matches getenv("X") / environ.get("X") / environ["X"] for an UPPER_SNAKE env var literal, with or
# without an ``os.`` prefix (so a ``from os import getenv`` alias is still caught). It cannot catch a
# read whose name is a *variable* rather than a literal (no static string scan can); the convention
# is to read env vars by literal, and this is enforced for every current guardrail.
_ENV_READ_RE = re.compile(
    r"""(?:os\.)?(?:getenv|environ\.get)\(\s*["']([A-Z_][A-Z0-9_]*)["']"""
    r"""|(?:os\.)?environ\[\s*["']([A-Z_][A-Z0-9_]*)["']""",
)

# A parameter whose name matches this is a credential and MUST be flagged ``secret`` — so a new
# guardrail that adds e.g. ``api_key`` without declaring it in ``_SECRET_PARAMS`` fails a test even
# though it is not in the hardcoded allowlist below. Kept in sync with ``_SECRET_PARAMS`` by
# ``test_credential_named_params_are_marked_secret`` (which asserts equality in both directions).
_CREDENTIAL_NAME_RE = re.compile(r"api_key|apikey|access_key|secret|password|api_client|boto3_session", re.IGNORECASE)


def _param_names(name: GuardrailName) -> set[str]:
    """Every create + validate parameter name in a guardrail's schema."""
    return {spec.name for spec in get_parameter_schema(name)}


def _env_vars_read_in_source(guardrail: str) -> set[str]:
    """The set of env-var name literals actually read anywhere in a guardrail's package."""
    found: set[str] = set()
    for path in (_GUARDRAILS_DIR / guardrail).rglob("*.py"):
        for match in _ENV_READ_RE.finditer(path.read_text(encoding="utf-8")):
            found.add(match.group(1) or match.group(2))
    return found


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


# --- Runtime-requirement declarations (env_var / secret / effectively_required / groups) ----------


@pytest.mark.parametrize("name", ALL_NAMES, ids=lambda n: n.value)
def test_runtime_declarations_reference_real_parameters(name: GuardrailName) -> None:
    """Every guardrail declared in a runtime-requirement map names only real, schema-listed params."""
    valid = _param_names(name)
    key = name.value
    declared_names = (
        set(generate_parameter_data._ENV_VAR_FALLBACKS.get(key, {}))
        | generate_parameter_data._SECRET_PARAMS.get(key, frozenset())
        | generate_parameter_data._EFFECTIVELY_REQUIRED.get(key, frozenset())
    )
    for group in generate_parameter_data._REQUIREMENT_GROUPS.get(key, []):
        declared_names |= set(group["parameters"])
    assert declared_names <= valid, f"{key}: declarations reference unknown params {declared_names - valid}"


@pytest.mark.parametrize("name", ALL_NAMES, ids=lambda n: n.value)
def test_env_var_and_effectively_required_imply_signature_optional(name: GuardrailName) -> None:
    """A value with an env-var fallback or an at-runtime requirement is optional *as an argument*."""
    for spec in get_parameter_schema(name):
        if spec.env_var is not None or spec.effectively_required:
            assert not spec.required, (
                f"{name.value}:{spec.name} is signature-required yet marks a fallback/runtime need"
            )


@pytest.mark.parametrize("name", ALL_NAMES, ids=lambda n: n.value)
def test_effectively_required_and_requirement_groups_are_disjoint(name: GuardrailName) -> None:
    """A param carries its requirement either via effectively_required or via a group, never both."""
    eff_required = {spec.name for spec in get_parameter_schema(name) if spec.effectively_required}
    grouped = {param for group in get_requirement_groups(name) for param in group.parameters}
    assert eff_required.isdisjoint(grouped), (
        f"{name.value}: {eff_required & grouped} is both effectively_required and grouped"
    )


@pytest.mark.parametrize("name", ALL_NAMES, ids=lambda n: n.value)
def test_requirement_groups_are_well_formed(name: GuardrailName) -> None:
    """Each group offers at least two ways to satisfy it (else it is just an effectively_required param)."""
    for group in get_requirement_groups(name):
        assert isinstance(group, RequirementGroup)
        choices = len(group.parameters) + len(group.env_vars)
        assert choices >= 2, f"{name.value}: single-choice group {group.parameters} should be effectively_required"
        assert group.description.strip(), f"{name.value}: requirement group missing a description"


@pytest.mark.parametrize("name", ALL_NAMES, ids=lambda n: n.value)
def test_env_var_declarations_match_guardrail_source(name: GuardrailName) -> None:
    """Bidirectional anti-drift: declared env vars are exactly those the guardrail actually reads.

    Catches a renamed/removed env var (declared but not read) and a newly-added env-var fallback that
    nobody declared (read but not declared) — the latter is how an env-backed requirement would
    silently escape the schema.
    """
    declared = set(generate_parameter_data._ENV_VAR_FALLBACKS.get(name.value, {}).values())
    read_in_source = _env_vars_read_in_source(name.value)
    assert declared == read_in_source, (
        f"{name.value}: env-var fallbacks declared={declared} but source reads={read_in_source}; "
        "update _ENV_VAR_FALLBACKS in scripts/generate_parameter_data.py and regenerate."
    )


def test_credentials_surface_as_secret_params() -> None:
    """The known credential parameters are present in the schema and flagged ``secret`` (not skipped).

    Guards the decision to stop dropping ``api_key`` from the schema: a config UI must see the
    credential (masked) rather than have it silently omitted.
    """
    expected_secret = {
        "alinia": {"api_key"},
        "azure_content_safety": {"api_key"},
        "azure_prompt_shields": {"api_key"},
        "bedrock_guardrails": {"aws_access_key_id", "aws_secret_access_key", "boto3_session"},
        "lakera_guard": {"api_key"},
        "openai_moderation": {"api_key"},
        "patronus": {"api_key"},
        "watsonx_guardian": {"api_key", "api_client"},
    }
    for guardrail, secrets in expected_secret.items():
        name = GuardrailName(guardrail)
        schema_secret = {spec.name for spec in get_parameter_schema(name) if spec.secret}
        assert schema_secret == secrets, f"{guardrail}: secret params {schema_secret} != expected {secrets}"


@pytest.mark.parametrize("name", ALL_NAMES, ids=lambda n: n.value)
def test_credential_named_params_are_marked_secret(name: GuardrailName) -> None:
    """Every credential-named param is ``secret`` and vice versa — catches a NEW unmasked credential.

    Unlike the hardcoded allowlist above, this runs over every guardrail, so a future guardrail that
    exposes e.g. ``api_key`` but forgets its ``_SECRET_PARAMS`` entry fails here (credential name but
    ``secret=False``). The reverse direction keeps ``_CREDENTIAL_NAME_RE`` honest: a ``secret`` param
    whose name the pattern misses (e.g. a new credential-bearing object) forces the pattern to grow.
    """
    specs = get_parameter_schema(name)
    credential_named = {spec.name for spec in specs if _CREDENTIAL_NAME_RE.search(spec.name)}
    marked_secret = {spec.name for spec in specs if spec.secret}
    assert credential_named == marked_secret, (
        f"{name.value}: credential-named params {credential_named} != secret-flagged {marked_secret}; "
        "add the missing param to _SECRET_PARAMS (or extend _CREDENTIAL_NAME_RE for a new credential shape)."
    )


def test_watsonx_api_client_satisfies_every_credential_group() -> None:
    """The ``api_client`` escape hatch is an alternative member of each watsonx credential group.

    Mirrors the runtime ``if api_client is None`` guard: supplying a pre-built client must satisfy
    the api_key, url, and project/space requirements at once.
    """
    groups = get_requirement_groups(GuardrailName.WATSONX_GUARDIAN)
    assert len(groups) == 3
    assert all("api_client" in group.parameters for group in groups)


def test_json_export_pairs_parameters_with_requirement_groups() -> None:
    """The committed JSON nests ``parameters`` + ``requirement_groups`` under each guardrail."""
    import json

    payload = json.loads(generate_parameters_json.DEFAULT_OUT.read_text(encoding="utf-8"))
    assert set(payload) == {n.value for n in GuardrailName}
    for guardrail, entry in payload.items():
        assert set(entry) == {"parameters", "requirement_groups"}, (
            f"{guardrail}: unexpected top-level keys {set(entry)}"
        )
        assert isinstance(entry["parameters"], list)
        assert isinstance(entry["requirement_groups"], list)
    watsonx = payload["watsonx_guardian"]["requirement_groups"]
    assert len(watsonx) == 3
