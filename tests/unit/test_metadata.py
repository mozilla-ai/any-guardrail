"""Parity and behavior tests for the guardrail taxonomy metadata (issue #182).

These tests guarantee the registry cannot silently drift as guardrails are added:
every ``GuardrailName`` has exactly one metadata entry, every guardrail class
mirrors that entry, the flavor-text ``description`` stays in sync with the class
docstring, and the recorded ``validate()`` kwargs match the real signatures.
"""

import ast
import inspect
import subprocess
import sys
from pathlib import Path

import pytest

import any_guardrail.taxonomy
from any_guardrail import AnyGuardrail, GuardrailName
from any_guardrail.base import Guardrail, ThreeStageGuardrail
from any_guardrail.registry import GUARDRAIL_METADATA
from any_guardrail.taxonomy import (
    BackendType,
    GuardrailCategory,
    GuardrailMetadata,
    GuardrailStage,
    OutputShape,
)

ALL_NAMES = list(GuardrailName)


def _guardrail_class(name: GuardrailName) -> type[Guardrail]:
    return AnyGuardrail._get_guardrail_class(name)


def _validate_signature_source(cls: type[Guardrail]) -> object:
    """Return the callable whose params define this guardrail's validate() inputs.

    A guardrail that overrides ``validate`` documents its own kwargs there; the
    ``StandardGuardrail`` classifiers inherit ``validate`` and instead take extra
    kwargs on ``_pre_processing``.
    """
    if "validate" in cls.__dict__:
        return cls.validate
    return getattr(cls, "_pre_processing")  # noqa: B009  # ThreeStageGuardrail-only attr


def _params_after_first(func: object) -> list[str]:
    """Named params after the first positional (the primary text arg), no *args/**kwargs."""
    sig = inspect.signature(func)  # type: ignore[arg-type]
    names = [
        pname
        for pname, param in sig.parameters.items()
        if pname not in ("self", "cls")
        and param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]
    return names[1:]


def test_registry_covers_all_guardrails_exactly() -> None:
    """Every GuardrailName has exactly one registry entry, and vice versa."""
    assert set(GUARDRAIL_METADATA) == set(GuardrailName)
    assert len(GUARDRAIL_METADATA) == len(ALL_NAMES)


@pytest.mark.parametrize("name", ALL_NAMES, ids=lambda n: n.value)
def test_class_metadata_is_registry_entry(name: GuardrailName) -> None:
    """Each class defines its own METADATA and it is (by identity) the registry entry."""
    cls = _guardrail_class(name)
    assert "METADATA" in cls.__dict__, f"{cls.__name__} does not set METADATA in its own body"
    assert cls.__dict__["METADATA"] is GUARDRAIL_METADATA[name]


@pytest.mark.parametrize("name", ALL_NAMES, ids=lambda n: n.value)
def test_description_matches_docstring_summary(name: GuardrailName) -> None:
    """description equals the class docstring's first line and reads as a sentence."""
    cls = _guardrail_class(name)
    doc = inspect.cleandoc(cls.__doc__ or "")
    first_line = doc.splitlines()[0] if doc else ""
    meta = GUARDRAIL_METADATA[name]
    assert meta.description == first_line, f"{cls.__name__}: metadata description != docstring first line"
    assert meta.description.endswith("."), f"{cls.__name__}: description should end with a period"


@pytest.mark.parametrize("name", ALL_NAMES, ids=lambda n: n.value)
def test_validate_kwargs_match_signature(name: GuardrailName) -> None:
    """Recorded kwargs partition the real signature; signature-required kwargs are marked required."""
    cls = _guardrail_class(name)
    source = _validate_signature_source(cls)
    after = set(_params_after_first(source))
    sig = inspect.signature(source)  # type: ignore[arg-type]
    sig_required = {
        pname
        for pname in after
        if sig.parameters[pname].default is inspect.Parameter.empty
        and sig.parameters[pname].kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    }

    meta = GUARDRAIL_METADATA[name]
    recorded = meta.required_validate_kwargs | meta.optional_validate_kwargs
    # required and optional are disjoint and together cover exactly the real params.
    assert meta.required_validate_kwargs.isdisjoint(meta.optional_validate_kwargs)
    assert recorded == after, f"{cls.__name__}: recorded kwargs {sorted(recorded)} != signature {sorted(after)}"
    # A kwarg the signature forces (no default) must be recorded as required.
    assert sig_required <= meta.required_validate_kwargs


@pytest.mark.parametrize("name", ALL_NAMES, ids=lambda n: n.value)
def test_primary_category_in_categories(name: GuardrailName) -> None:
    """The headline category is always one of the guardrail's categories."""
    meta = GUARDRAIL_METADATA[name]
    assert meta.primary_category in meta.categories


@pytest.mark.parametrize("name", ALL_NAMES, ids=lambda n: n.value)
def test_default_license_non_empty(name: GuardrailName) -> None:
    """Every guardrail records a non-empty default_license (issue #211)."""
    assert GUARDRAIL_METADATA[name].default_license.strip(), f"{name.value}: default_license must be non-empty"


@pytest.mark.parametrize("name", ALL_NAMES, ids=lambda n: n.value)
def test_variant_licenses_reference_supported_models(name: GuardrailName) -> None:
    """Each per-variant license points at a real SUPPORTED_MODELS entry and names a license (issue #211)."""
    meta = GUARDRAIL_METADATA[name]
    if not meta.variant_licenses:
        return
    supported = set(_guardrail_class(name).SUPPORTED_MODELS)
    seen: set[str] = set()
    for variant in meta.variant_licenses:
        assert variant.model_id in supported, f"{name.value}: variant {variant.model_id!r} not in SUPPORTED_MODELS"
        assert variant.license.strip(), f"{name.value}: variant {variant.model_id!r} has an empty license"
        assert variant.model_id not in seen, f"{name.value}: duplicate variant {variant.model_id!r}"
        seen.add(variant.model_id)


def test_variant_licenses_serialize_sorted_by_model_id() -> None:
    """The variant_licenses JSON export is deterministically sorted by model_id (issue #211)."""
    for name in ALL_NAMES:
        dumped = GUARDRAIL_METADATA[name].model_dump(mode="json")["variant_licenses"]
        model_ids = [entry["model_id"] for entry in dumped]
        assert model_ids == sorted(model_ids), f"{name.value}: variant_licenses not sorted by model_id"


def test_metadata_is_frozen() -> None:
    """GuardrailMetadata instances are immutable."""
    meta = next(iter(GUARDRAIL_METADATA.values()))
    with pytest.raises((TypeError, ValueError)):
        meta.vendor = "changed"


def test_primary_not_in_categories_rejected() -> None:
    """Constructing metadata whose primary_category is absent from categories fails."""
    with pytest.raises(ValueError, match="primary_category"):
        GuardrailMetadata(
            description="X — y.",
            display_name="X",
            categories=frozenset({GuardrailCategory.CONTENT_SAFETY}),
            primary_category=GuardrailCategory.PII,
            stages=frozenset({GuardrailStage.INPUT}),
            output_shapes=frozenset({OutputShape.BINARY}),
            backend=BackendType.LOCAL_ENCODER,
            vendor="X",
            default_license="apache-2.0",
        )


def test_metadata_query_loads_no_guardrail_modules() -> None:
    """Filtering/grouping runs off the registry without importing guardrail backends.

    The registry keeps queries cheap by never importing the 38 implementation
    modules (each of which can pull heavy, model-specific dependencies). Verified in
    a fresh interpreter so earlier test imports don't mask a regression.
    """
    code = (
        "import sys\n"
        "from any_guardrail import AnyGuardrail, GuardrailCategory, BackendType\n"
        "AnyGuardrail.list_guardrails(category=GuardrailCategory.PROMPT_INJECTION, backend=BackendType.LOCAL_ENCODER)\n"
        "AnyGuardrail.group_by('category')\n"
        "AnyGuardrail.metadata(next(iter(__import__('any_guardrail').GuardrailName)))\n"
        "impl = [m for m in sys.modules if m.startswith('any_guardrail.guardrails.')]\n"
        "assert impl == [], impl\n"
        "print('ok')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)  # noqa: S603
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


def test_taxonomy_module_is_leaf() -> None:
    """taxonomy.py depends only on the stdlib and pydantic.

    This is what keeps the registry (and therefore queries) genuinely cheap: the
    metadata model and its enums never reach into ``any_guardrail`` internals or a
    model backend. Asserted statically on the source so it holds regardless of what
    the package ``__init__`` eagerly imports (it pulls in providers, hence torch).
    """
    allowed_roots = {"enum", "typing", "pydantic"}
    source = Path(any_guardrail.taxonomy.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    assert roots <= allowed_roots, f"taxonomy.py imports beyond stdlib/pydantic: {sorted(roots - allowed_roots)}"


def test_list_guardrails_no_filter_returns_all() -> None:
    """An unfiltered listing returns every guardrail in declaration order."""
    assert AnyGuardrail.list_guardrails() == ALL_NAMES


def test_list_guardrails_and_semantics_across_dimensions() -> None:
    """Filters AND together: category ∩ backend narrows the result."""
    pi = set(AnyGuardrail.list_guardrails(category=GuardrailCategory.PROMPT_INJECTION))
    encoders = set(AnyGuardrail.list_guardrails(backend=BackendType.LOCAL_ENCODER))
    combined = set(
        AnyGuardrail.list_guardrails(category=GuardrailCategory.PROMPT_INJECTION, backend=BackendType.LOCAL_ENCODER)
    )
    assert combined == pi & encoders
    assert combined  # non-empty: several encoder injection classifiers exist


def test_list_guardrails_category_any_overlap_with_iterable() -> None:
    """A set-valued filter matches guardrails carrying ANY of the requested values."""
    pi = set(AnyGuardrail.list_guardrails(category=GuardrailCategory.PROMPT_INJECTION))
    hall = set(AnyGuardrail.list_guardrails(category=GuardrailCategory.HALLUCINATION))
    union = set(
        AnyGuardrail.list_guardrails(category=[GuardrailCategory.PROMPT_INJECTION, GuardrailCategory.HALLUCINATION])
    )
    assert union == pi | hall


def test_list_guardrails_scalar_flags() -> None:
    """requires_api_key selects exactly the hosted-API guardrails."""
    api_key = set(AnyGuardrail.list_guardrails(requires_api_key=True))
    hosted = {n for n in ALL_NAMES if GUARDRAIL_METADATA[n].requires_api_key}
    assert api_key == hosted
    assert api_key  # non-empty


def test_group_by_covers_every_guardrail() -> None:
    """Every guardrail appears at least once under a set-valued grouping."""
    groups = AnyGuardrail.group_by("category")
    covered = {name for names in groups.values() for name in names}
    assert covered == set(ALL_NAMES)
    # keys are sorted category values
    assert list(groups) == sorted(groups)


def test_group_by_scalar_dimension() -> None:
    """Grouping by a scalar dimension partitions the guardrails."""
    groups = AnyGuardrail.group_by("backend")
    counts = {k: len(v) for k, v in groups.items()}
    assert sum(counts.values()) == len(ALL_NAMES)
    assert set(groups) <= {b.value for b in BackendType}


def test_group_by_unknown_dimension_raises() -> None:
    """An unsupported grouping dimension is a clear error."""
    with pytest.raises(ValueError, match="Unknown grouping dimension"):
        AnyGuardrail.group_by("nonsense")


def test_metadata_lookup_returns_registry_entry() -> None:
    """AnyGuardrail.metadata returns the canonical registry object."""
    for name in ALL_NAMES:
        assert AnyGuardrail.metadata(name) is GUARDRAIL_METADATA[name]


def test_standard_guardrails_use_inherited_validate() -> None:
    """Sanity: the classifiers we treat as inheriting validate really do (guards the kwarg test)."""
    cls = _guardrail_class(GuardrailName.PROTECTAI)
    assert "validate" not in cls.__dict__
    assert issubclass(cls, ThreeStageGuardrail)


def test_output_shapes_score_signal_accuracy() -> None:
    """Regression test for issue #225.

    ``OutputShape.SCORE``/``RUBRIC`` membership is the queryable contract for
    whether ``GuardrailOutput.score`` is ever populated. These six entries had
    drifted from actual guardrail behavior (confirmed against each guardrail's
    ``GuardrailOutput(score=...)`` call sites); pin the corrected values so the
    drift can't silently reappear.
    """
    expected_has_score = {
        GuardrailName.GRANITE_GUARDIAN: False,  # only ever emits a categorical yes/no verdict
        GuardrailName.QWEN3_GUARD_STREAM: True,
        GuardrailName.LAKERA_GUARD: True,
        GuardrailName.WATSONX_GUARDIAN: True,
        GuardrailName.AZURE_PROMPT_SHIELDS: True,
        GuardrailName.BEDROCK_GUARDRAILS: True,
    }
    for name, has_score in expected_has_score.items():
        output_shapes = GUARDRAIL_METADATA[name].output_shapes
        declares_score = bool({OutputShape.SCORE, OutputShape.RUBRIC} & output_shapes)
        assert declares_score == has_score, (
            f"{name.value}: expected SCORE/RUBRIC membership {has_score}, got {output_shapes}"
        )


def _metadata(**overrides: object) -> GuardrailMetadata:
    """Build a minimal valid GuardrailMetadata, overriding individual fields."""
    fields: dict[str, object] = {
        "description": "X — y.",
        "display_name": "X",
        "categories": frozenset({GuardrailCategory.PROMPT_INJECTION}),
        "primary_category": GuardrailCategory.PROMPT_INJECTION,
        "stages": frozenset({GuardrailStage.INPUT}),
        "output_shapes": frozenset({OutputShape.BINARY}),
        "backend": BackendType.LOCAL_ENCODER,
        "vendor": "X",
        "default_license": "apache-2.0",
    }
    fields.update(overrides)
    return GuardrailMetadata(**fields)  # type: ignore[arg-type]


def test_alternate_backends_defaults_to_empty() -> None:
    """Only guardrails whose alternates cross a BackendType boundary declare any."""
    assert _metadata().alternate_backends == frozenset()
    declared = {name for name in ALL_NAMES if GUARDRAIL_METADATA[name].alternate_backends}
    assert declared == {GuardrailName.SUSFACTOR}


def test_susfactor_declares_its_hosted_alternate() -> None:
    """SusFactor's gated local model is also reachable through 0DIN's hosted API."""
    meta = GUARDRAIL_METADATA[GuardrailName.SUSFACTOR]
    assert meta.backend == BackendType.LOCAL_ENCODER
    assert meta.alternate_backends == frozenset({BackendType.HOSTED_API})


def test_backend_repeated_in_alternate_backends_rejected() -> None:
    """alternate_backends must list genuine alternatives, not restate `backend`."""
    with pytest.raises(ValueError, match="alternate_backends"):
        _metadata(
            backend=BackendType.LOCAL_ENCODER,
            alternate_backends=frozenset({BackendType.LOCAL_ENCODER}),
        )


def test_alternate_backends_serialize_as_a_sorted_string_list() -> None:
    """Set-valued fields serialize deterministically so the JSON export is stable."""
    meta = _metadata(alternate_backends=frozenset({BackendType.LOCAL_DECODER, BackendType.HOSTED_API}))

    assert meta.model_dump()["alternate_backends"] == ["hosted_api", "local_decoder"]


def test_backend_grouping_still_partitions_every_guardrail() -> None:
    """alternate_backends is metadata only: it must not leak into backend grouping."""
    groups = AnyGuardrail.group_by("backend")

    assert sum(len(names) for names in groups.values()) == len(ALL_NAMES)
    assert GuardrailName.SUSFACTOR not in groups.get(BackendType.HOSTED_API.value, [])
