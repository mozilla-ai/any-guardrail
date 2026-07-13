"""Tests for the guardrail authored-content registry (issues #20 / #87).

Verify the per-kind API, that content mirrored from a live source stays byte-identical (drift
tests), that ``content.py`` is a dependency-free leaf, and that content queries never import a
guardrail backend.
"""

import ast
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

import any_guardrail.content
from any_guardrail import AnyGuardrail, GuardrailName
from any_guardrail.content import AuthoredContent, ContentKind
from any_guardrail.content_registry import CONTENT_BEARING, CONTENT_REGISTRY

CONTENT_NAMES = sorted(CONTENT_BEARING, key=lambda n: n.value)


def test_registry_covers_content_bearing_exactly() -> None:
    """CONTENT_BEARING mirrors the registry keys, all valid guardrail names."""
    assert set(CONTENT_REGISTRY) == CONTENT_BEARING
    assert CONTENT_BEARING <= set(GuardrailName)


@pytest.mark.parametrize("name", CONTENT_NAMES, ids=lambda n: n.value)
def test_keys_unique_within_kind(name: GuardrailName) -> None:
    """A guardrail's content keys are unique within each kind (so get_X(name, key) is unambiguous)."""
    seen: set[tuple[str, str]] = set()
    for c in CONTENT_REGISTRY[name]:
        pair = (c.kind.value, c.key)
        assert pair not in seen, f"{name.value}: duplicate {pair}"
        seen.add(pair)


@pytest.mark.parametrize("name", CONTENT_NAMES, ids=lambda n: n.value)
def test_per_kind_api_round_trips(name: GuardrailName) -> None:
    """list_* returns the sorted keys and get_* returns each item's content, per kind."""
    lists = {
        ContentKind.POLICY: AnyGuardrail.list_policies(name),
        ContentKind.RUBRIC: AnyGuardrail.list_rubrics(name),
        ContentKind.CRITERIA: AnyGuardrail.list_criteria(name),
    }
    getters = {
        ContentKind.POLICY: AnyGuardrail.get_policy,
        ContentKind.RUBRIC: AnyGuardrail.get_rubric,
        ContentKind.CRITERIA: AnyGuardrail.get_criteria,
    }
    for kind, keys in lists.items():
        assert keys == sorted(keys)
        for key in keys:
            content = getters[kind](name, key)
            match = next(c for c in CONTENT_REGISTRY[name] if c.kind == kind and c.key == key)
            assert content == match.content
    # every registered item is reachable through exactly one kind's list
    reachable = sum(len(v) for v in lists.values())
    assert reachable == len(CONTENT_REGISTRY[name])


def test_get_missing_raises_keyerror() -> None:
    """Unknown (name, kind, key) is a clear error; non-content guardrails list nothing."""
    with pytest.raises(KeyError):
        AnyGuardrail.get_policy(GuardrailName.SHIELD_GEMMA, "does-not-exist")
    assert AnyGuardrail.list_criteria(GuardrailName.DEEPSET) == []
    assert AnyGuardrail.list_policies(GuardrailName.DEEPSET) == []


def test_content_frozen() -> None:
    """AuthoredContent instances are immutable."""
    item = CONTENT_REGISTRY[GuardrailName.GRANITE_GUARDIAN][0]
    with pytest.raises((TypeError, ValidationError)):
        item.content = "changed"


def test_content_module_is_leaf() -> None:
    """content.py depends only on the stdlib and pydantic (keeps content queries cheap)."""
    allowed_roots = {"enum", "typing", "pydantic"}
    source = Path(any_guardrail.content.__file__).read_text(encoding="utf-8")
    roots: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    assert roots <= allowed_roots, f"content.py imports beyond stdlib/pydantic: {sorted(roots - allowed_roots)}"


def test_content_query_loads_no_guardrail_modules() -> None:
    """The content API runs off the registry without importing any guardrail backend."""
    code = (
        "import sys\n"
        "from any_guardrail import AnyGuardrail, GuardrailName\n"
        "AnyGuardrail.get_criteria(GuardrailName.GRANITE_GUARDIAN, 'harm')\n"
        "AnyGuardrail.list_policies(GuardrailName.SHIELD_GEMMA)\n"
        "impl = [m for m in sys.modules if m.startswith('any_guardrail.guardrails.')]\n"
        "assert impl == [], impl\n"
        "print('ok')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)  # noqa: S603
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


# --------------------------------------------------------------------------- #
# Drift: content mirrored from a live source stays byte-identical to it.        #
# --------------------------------------------------------------------------- #


def test_granite_criteria_match_source() -> None:
    """Granite criteria in the registry equal GraniteGuardianRisk (the in-repo source)."""
    from any_guardrail.guardrails.granite_guardian.granite_guardian import GraniteGuardianRisk

    for key in AnyGuardrail.list_criteria(GuardrailName.GRANITE_GUARDIAN):
        assert AnyGuardrail.get_criteria(GuardrailName.GRANITE_GUARDIAN, key) == getattr(GraniteGuardianRisk, key.upper())


def test_flowjudge_content_matches_library() -> None:
    """Flow-Judge criteria and rubrics equal the installed flow_judge presets."""
    pytest.importorskip("flow_judge")
    import flow_judge.metrics as metrics
    from flow_judge.utils.prompt_formatter import format_rubric

    for key in AnyGuardrail.list_criteria(GuardrailName.FLOWJUDGE):
        assert AnyGuardrail.get_criteria(GuardrailName.FLOWJUDGE, key) == getattr(metrics, key.upper()).criteria
    for key in AnyGuardrail.list_rubrics(GuardrailName.FLOWJUDGE):
        assert AnyGuardrail.get_rubric(GuardrailName.FLOWJUDGE, key) == format_rubric(getattr(metrics, key.upper()).rubric)


def test_authored_content_data_matches_live_sources() -> None:
    """The generated data module is byte-identical to the current Granite / flow_judge sources."""
    pytest.importorskip("flow_judge")
    import flow_judge.metrics as metrics
    from flow_judge.utils.prompt_formatter import format_rubric

    from any_guardrail import _authored_content_data as data
    from any_guardrail.guardrails.granite_guardian.granite_guardian import GraniteGuardianRisk

    granite = {a.lower(): getattr(GraniteGuardianRisk, a) for a in vars(GraniteGuardianRisk) if a.isupper()}
    presets = [n for n in dir(metrics) if n.isupper() and not n.startswith("_")]
    assert data.GRANITE_CRITERIA == granite
    assert data.FLOWJUDGE_CRITERIA == {n.lower(): getattr(metrics, n).criteria for n in presets}
    assert data.FLOWJUDGE_RUBRICS == {n.lower(): format_rubric(getattr(metrics, n).rubric) for n in presets}
