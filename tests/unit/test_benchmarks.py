"""Tests for the benchmark model-card registry + comparability guard (issue #194).

The load-bearing tests are the adversarial cohort fixtures: the renderer must refuse to treat the
classic misleading comparisons (ToxicChat revisions, Optimal-F1 vs F1@0.5, AUC-in-F1-column,
cross-harness baselines, missing-as-zero, missing provenance) as comparable.
"""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

import any_guardrail._benchmark_data as benchmark_data_module
import any_guardrail.benchmarks as benchmarks_module
from any_guardrail import (
    BenchmarkResult,
    BenchmarkSource,
    BenchmarkSourceKind,
    ComparisonCohort,
    GuardrailCategory,
    GuardrailName,
)
from any_guardrail.benchmark_registry import BENCHMARK_REGISTRY, get_benchmarks, group_comparable

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

import generate_api_docs
import generate_benchmark_schema
import generate_benchmarks_json


def _result(
    *,
    dataset: str = "ToxicChat",
    revision: str = "1123",
    metric: str = "f1",
    threshold: str = "f1@0.5",
    harness: str = "qwen3guard-report",
    value: float | None = 0.8,
    contamination: bool = False,
) -> BenchmarkResult:
    return BenchmarkResult(
        guardrail="llama_guard",
        category="content_safety",
        value=value,
        source=BenchmarkSource(kind=BenchmarkSourceKind.PUBLISHED, url="https://example.com/card"),
        contamination=contamination,
        cohort=ComparisonCohort(
            dataset=dataset,
            dataset_revision=revision,
            label_mapping="unsafe=positive",
            metric=metric,
            threshold_policy=threshold,
            harness=harness,
        ),
    )


# --- registry -----------------------------------------------------------------


def test_registry_covers_all_guardrails_and_ships_empty() -> None:
    """Every GuardrailName has an entry, and the committed registry is empty (a ready-to-fill skeleton)."""
    assert set(BENCHMARK_REGISTRY) == set(GuardrailName)
    assert sum(len(results) for results in BENCHMARK_REGISTRY.values()) == 0


@pytest.mark.parametrize("name", list(GuardrailName), ids=lambda n: n.value)
def test_get_benchmarks_returns_list(name: GuardrailName) -> None:
    assert get_benchmarks(name) == []


# --- provenance ---------------------------------------------------------------


def test_published_source_requires_url() -> None:
    with pytest.raises(ValueError, match="published"):
        BenchmarkSource(kind=BenchmarkSourceKind.PUBLISHED)


def test_measured_source_requires_harness_version() -> None:
    with pytest.raises(ValueError, match="measured"):
        BenchmarkSource(kind=BenchmarkSourceKind.MEASURED)


# --- the comparability invariant (adversarial fixtures) -----------------------


def test_toxicchat_revisions_are_not_comparable() -> None:
    groups = group_comparable([_result(revision="1123"), _result(revision="0124")])
    assert len(groups) == 2, "ToxicChat 1123 and 0124 must not share a cohort"


def test_optimal_f1_and_f1_at_half_are_not_comparable() -> None:
    groups = group_comparable([_result(threshold="f1@0.5"), _result(threshold="optimal-f1")])
    assert len(groups) == 2


def test_auc_and_f1_are_not_comparable() -> None:
    groups = group_comparable([_result(metric="f1", threshold="f1@0.5"), _result(metric="auc", threshold="n/a")])
    assert len(groups) == 2


def test_cross_harness_baselines_are_not_comparable() -> None:
    groups = group_comparable([_result(harness="qwen3guard-report"), _result(harness="injecguard-paper")])
    assert len(groups) == 2


def test_same_cohort_scores_are_grouped_together() -> None:
    groups = group_comparable([_result(value=0.8), _result(value=0.7)])
    assert len(groups) == 1
    assert len(next(iter(groups.values()))) == 2


def test_missing_value_stays_none_not_zero() -> None:
    result = _result(value=None)
    assert result.value is None
    assert generate_api_docs._benchmark_result_table([result]).count("—") == 1
    assert "| 0 |" not in generate_api_docs._benchmark_result_table([result])


# --- rendering ----------------------------------------------------------------


def test_benchmark_table_shows_provenance_contamination_and_cohort_keys() -> None:
    table = generate_api_docs._benchmark_result_table(
        [_result(value=0.912, contamination=True, revision="0124", harness="qwen3-report")]
    )
    assert "0.912" in table
    assert "⚠️" in table  # contamination flag
    assert "[published](https://example.com/card)" in table
    assert "ToxicChat (0124)" in table  # dataset + revision visible
    assert "qwen3-report" in table  # harness visible


def test_empty_benchmarks_section_renders_a_note() -> None:
    section = generate_api_docs._benchmarks_section(GuardrailName.LLAMA_GUARD)
    assert "## Benchmarks" in section
    assert "No benchmark results recorded yet" in section


def test_license_section_renders_variant_table() -> None:
    section = generate_api_docs._license_section(GuardrailName.LLAMA_GUARD)
    assert "## License" in section
    assert "Meta" in section
    assert "meta-llama/Llama-Guard-4-12B" in section  # per-variant licenses (from #211)


# --- parity + leaf invariants -------------------------------------------------


def test_benchmarks_json_matches_registry() -> None:
    expected = generate_benchmarks_json.render(generate_benchmarks_json.build_payload())
    assert generate_benchmarks_json.DEFAULT_OUT.read_text(encoding="utf-8") == expected


def test_benchmark_schema_matches_models() -> None:
    expected = generate_benchmark_schema.render(generate_benchmark_schema.build_schema())
    assert generate_benchmark_schema.DEFAULT_OUT.read_text(encoding="utf-8") == expected


def test_committed_results_reference_valid_names_and_categories() -> None:
    valid_names = {n.value for n in GuardrailName}
    valid_categories = {c.value for c in GuardrailCategory}
    for results in BENCHMARK_REGISTRY.values():
        for result in results:
            assert result.guardrail in valid_names
            assert result.category in valid_categories


def _import_roots(module_file: str) -> set[str]:
    tree = ast.parse(Path(module_file).read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_benchmarks_module_is_leaf() -> None:
    roots = _import_roots(benchmarks_module.__file__)
    assert roots <= {"enum", "typing", "pydantic"}, f"benchmarks.py imports beyond stdlib/pydantic: {roots}"


def test_benchmark_data_module_only_imports_benchmarks() -> None:
    roots = _import_roots(benchmark_data_module.__file__)
    assert roots <= {"any_guardrail"}, f"_benchmark_data.py imports beyond any_guardrail: {roots}"


def test_reading_benchmarks_loads_no_guardrail_modules() -> None:
    code = (
        "import sys\n"
        "from any_guardrail.benchmark_registry import get_benchmarks\n"
        "from any_guardrail import GuardrailName\n"
        "for n in GuardrailName:\n"
        "    get_benchmarks(n)\n"
        "impl = [m for m in sys.modules if m.startswith('any_guardrail.guardrails.')]\n"
        "assert impl == [], impl\n"
        "print('ok')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)  # noqa: S603
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout
