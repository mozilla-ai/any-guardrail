"""Generate API reference documentation from Python source code.

Extracts docstrings and signatures from any-guardrail source and writes
GitBook-compatible Markdown pages to a given output directory.

When called from convert_to_gitbook.py, output goes to site/api/ so that
the old mkdocstrings stubs in docs/api/ never enter the GitBook output.

Usage:
    python scripts/generate_api_docs.py                  # writes to docs/api/
    python scripts/generate_api_docs.py --out site/api   # writes to site/api/
"""

from __future__ import annotations

import argparse
import ast
import inspect
import re
import sys
import textwrap
from pathlib import Path
from typing import Any

# Ensure the package is importable when run directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

DEFAULT_OUT = Path(__file__).parent.parent / "docs" / "api"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MODULE_PREFIXES = [
    "any_guardrail.types.",
    "any_guardrail.base.",
    "any_guardrail.guardrails.",
    "any_guardrail.providers.base.",
    "any_guardrail.",
    "collections.abc.",
    "pydantic.main.",
    "pydantic.",
    "typing.",
    "typing_extensions.",
    "builtins.",
]


def _clean_type_str(text: str) -> str:
    return re.sub(r"[\w.]+\.([A-Z]\w*)", r"\1", text)


def _format_annotation(annotation: Any) -> str:
    if annotation is inspect.Parameter.empty:
        return ""
    try:
        if isinstance(annotation, type):
            return annotation.__name__
        text = str(annotation)
        text = text.replace("typing.", "").replace("typing_extensions.", "")
        return _clean_type_str(text)
    except Exception:
        return str(annotation)


_MAX_DEFAULT_LENGTH = 60


def _format_default(default: Any) -> str:
    if default is inspect.Parameter.empty:
        return ""
    if default is None:
        return "None"
    if isinstance(default, str):
        # Defaults are rendered inside one Markdown table cell: collapse
        # newlines/runs of whitespace, escape pipes, and truncate long prompts.
        rendered = " ".join(default.split()).replace("|", "\\|")
        if len(rendered) > _MAX_DEFAULT_LENGTH:
            rendered = rendered[: _MAX_DEFAULT_LENGTH - 1] + "…"
        return f'"{rendered}"'
    return repr(default)


def _parse_args_section(doc: str | None) -> dict[str, str]:
    """Extract per-parameter descriptions from a Google-style ``Args:`` section.

    Returns a mapping of parameter name to its full (continuation-joined)
    description, so the signature tables can carry the documented input space
    instead of dropping it.
    """
    if not doc:
        return {}
    text = textwrap.dedent(doc)
    match = re.search(
        r"\n\s*Args\s*:\s*\n(.*?)(?=\n\s*(?:Returns|Raises|Note|Example|Examples|Yields|Attributes)\s*:|\Z)",
        text,
        re.DOTALL,
    )
    if not match:
        return {}
    block = textwrap.dedent(match.group(1))
    params: dict[str, str] = {}
    current: str | None = None
    for line in block.splitlines():
        entry = (
            re.match(r"^(\*{0,2}\w+)\s*(?:\([^)]*\))?\s*:\s+(.*)$", line) if not line.startswith((" ", "\t")) else None
        )
        if entry:
            current = entry.group(1).lstrip("*")
            params[current] = entry.group(2).strip()
        elif current and line.strip():
            params[current] += " " + line.strip()
    return params


def _table_cell(text: str) -> str:
    """Collapse whitespace and escape pipes so text fits in one Markdown table cell."""
    return " ".join(text.split()).replace("|", "\\|")


def _sig_table(func: Any, skip_self: bool = True, param_docs: dict[str, str] | None = None) -> str:
    try:
        sig = inspect.signature(func)
    except (ValueError, TypeError):
        return ""

    param_docs = param_docs or {}
    rows = []
    documented = False
    for name, param in sig.parameters.items():
        if skip_self and name in ("self", "cls"):
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        ann = _format_annotation(param.annotation)
        default = _format_default(param.default)
        required = "Yes" if param.default is inspect.Parameter.empty else "No"
        default_cell = f"`{default}`" if default else "—"
        ann_cell = f"`{ann}`" if ann else "—"
        doc_cell = _table_cell(param_docs.get(name, "")) or "—"
        documented = documented or doc_cell != "—"
        rows.append((f"| `{name}` | {ann_cell} | {required} | {default_cell} |", f" {doc_cell} |"))

    if not rows:
        return ""
    if documented:
        header = (
            "| Parameter | Type | Required | Default | Description |\n"
            "|-----------|------|----------|---------|-------------|\n"
        )
        return header + "\n".join(base + desc for base, desc in rows)
    header = "| Parameter | Type | Required | Default |\n|-----------|------|----------|---------|\n"
    return header + "\n".join(base for base, _ in rows)


def _return_annotation(func: Any) -> str:
    try:
        sig = inspect.signature(func)
        ann = sig.return_annotation
        if ann is inspect.Parameter.empty:
            return ""
        return _format_annotation(ann)
    except Exception:
        return ""


def _clean_docstring(doc: str | None) -> str:
    if not doc:
        return ""
    return textwrap.dedent(doc).strip()


def _doc_summary(doc: str | None) -> str:
    """Return only the description portion of a Google-style docstring.

    Strips Args:, Returns:, Raises:, Note:, Example: sections so they don't
    appear as raw text alongside the generated parameter table.
    """
    if not doc:
        return ""
    text = textwrap.dedent(doc).strip()
    # Split at the first Google-style section header
    summary = re.split(r"\n\s*(?:Args|Returns|Raises|Note|Example|Examples)\s*:", text)[0]
    return summary.strip()


def _strip_args_section(doc: str) -> str:
    """Remove only the ``Args:`` block from a docstring, keeping every other section.

    Class docstrings that document constructor args render next to the generated
    constructor table (which now carries those descriptions), so the raw block
    would be duplicated.
    """
    stripped = re.sub(
        r"\n\s*Args\s*:\s*\n.*?(?=\n\s*(?:Returns|Raises|Note|Example|Examples|Yields|Attributes)\s*:|\Z)",
        "\n",
        doc,
        flags=re.DOTALL,
    )
    return stripped.strip()


def _section(title: str, level: int = 2) -> str:
    return f"{'#' * level} {title}\n"


# ---------------------------------------------------------------------------
# Model-card sections (issue #194): benchmarks + license
# ---------------------------------------------------------------------------


def _guardrail_name_from_module(module_path: str) -> Any:
    """Derive the ``GuardrailName`` from a guardrail module path, or ``None`` if it isn't one."""
    from any_guardrail.base import GuardrailName

    try:
        return GuardrailName(module_path.rsplit(".", 1)[-1])
    except ValueError:
        return None


def _benchmark_result_table(results: list[Any]) -> str:
    """Render a guardrail's benchmark results as a Markdown table (pure; used by the page + tests).

    One row per result, grouped by category. A missing value renders as ``—`` (never ``0``);
    a contamination flag adds ``⚠️``; provenance and the comparison-cohort keys (dataset revision,
    metric, threshold policy, harness) are shown so scores are never silently treated as comparable.
    """
    from any_guardrail.benchmarks import BenchmarkSourceKind

    by_category: dict[str, list[Any]] = {}
    for result in results:
        by_category.setdefault(result.category, []).append(result)

    lines: list[str] = []
    for category in sorted(by_category):
        lines.append(_section(category.replace("_", " ").title(), level=3))
        lines.append("| Dataset (rev) | Metric | Threshold | Value | Harness | Source | Contam. |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for result in by_category[category]:
            cohort = result.cohort
            value = "—" if result.value is None else f"{result.value:g}"
            if result.source.kind is BenchmarkSourceKind.PUBLISHED:
                source = f"[published]({result.source.url})" if result.source.url else "published"
            else:
                source = f"measured:{result.source.harness_version}"
            contam = "⚠️" if result.contamination else ""
            lines.append(
                f"| {_table_cell(f'{cohort.dataset} ({cohort.dataset_revision})')} "
                f"| {_table_cell(cohort.metric)} | {_table_cell(cohort.threshold_policy)} "
                f"| {value} | {_table_cell(cohort.harness)} | {source} | {contam} |"
            )
        lines.append("")
    return "\n".join(lines)


def _benchmarks_section(gname: Any) -> str:
    """Render the ``## Benchmarks`` section for a guardrail (empty note when no results)."""
    try:
        from any_guardrail.benchmark_registry import get_benchmarks

        results = get_benchmarks(gname)
        lines = [_section("Benchmarks")]
        if not results:
            lines.append(
                "No benchmark results recorded yet. See the [benchmark methodology](../../benchmarks.md) "
                "for how numbers are harvested (published) or measured and added.\n"
            )
            return "\n".join(lines)
        lines.append(_benchmark_result_table(results))
        return "\n".join(lines)
    except Exception:
        return ""


def _license_section(gname: Any) -> str:
    """Render the ``## License`` section for a guardrail from the taxonomy metadata."""
    try:
        from any_guardrail.registry import GUARDRAIL_METADATA

        meta = GUARDRAIL_METADATA[gname]
        lines = [
            _section("License"),
            f"- **Vendor:** {meta.vendor}",
            f"- **Default license:** `{meta.default_license}` (of the default model/service)",
        ]
        if meta.variant_licenses:
            lines.append("")
            lines.append("| Model variant | License |")
            lines.append("| --- | --- |")
            lines.extend(
                f"| `{variant.model_id}` | `{variant.license}` |"
                for variant in sorted(meta.variant_licenses, key=lambda v: v.model_id)
            )
        lines.append("")
        return "\n".join(lines)
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Page generators
# ---------------------------------------------------------------------------


def _guardrail_page(module_path: str, class_name: str) -> str:
    import importlib

    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)

    lines: list[str] = [f"# {class_name}\n"]

    class_doc = _clean_docstring(inspect.getdoc(cls))
    class_args = _parse_args_section(class_doc)
    if class_doc:
        lines.append(_strip_args_section(class_doc) + "\n")

    supported = getattr(cls, "SUPPORTED_MODELS", [])
    if supported:
        lines.append(_section("Supported Models"))
        for m in supported:
            lines.append(f"- `{m}`")
        lines.append("")

    init = getattr(cls, "__init__", None)
    if init:
        init_doc = _doc_summary(inspect.getdoc(init))
        lines.append(_section("Constructor"))
        # Constructor args may be documented on the class docstring, __init__, or both.
        table = _sig_table(init, param_docs={**class_args, **_parse_args_section(inspect.getdoc(init))})
        if table:
            lines.append(table + "\n")
        if init_doc and init_doc != class_doc:
            lines.append(init_doc + "\n")

    validate = getattr(cls, "validate", None)
    if validate and not getattr(validate, "__isabstractmethod__", False):
        lines.append(_section("validate"))
        val_doc = _doc_summary(inspect.getdoc(validate))
        if val_doc:
            lines.append(val_doc + "\n")
        table = _sig_table(validate, param_docs=_parse_args_section(inspect.getdoc(validate)))
        if table:
            lines.append("**Parameters**\n")
            lines.append(table + "\n")
        ret = _return_annotation(validate)
        if ret:
            lines.append(f"**Returns:** `{ret}`\n")

    # Model-card sections (issue #194): benchmark evidence + license, derived from the registries.
    gname = _guardrail_name_from_module(module_path)
    if gname is not None:
        for section in (_benchmarks_section(gname), _license_section(gname)):
            if section:
                lines.append(section)

    return "\n".join(lines)


# Extras whose name can't be derived as `class_name.lower()` (e.g. guardrails
# that reuse another backend's extra).
_INSTALL_EXTRA_OVERRIDES = {
    "AzureContentSafety": "azure-content-safety",
    "AzurePromptShields": "azure-content-safety",
    "WatsonxGuardian": "watsonx",
}


def _stub_page(class_name: str) -> str:
    """Minimal page for a guardrail whose heavy deps aren't installed."""
    extra = _INSTALL_EXTRA_OVERRIDES.get(class_name, class_name.lower())
    return (
        f"# {class_name}\n\n"
        "> API reference for this guardrail requires its optional dependencies to be installed.\n\n"
        f"Install with: `pip install 'any-guardrail[{extra}]'`\n"
    )


def _provider_page(module_path: str, class_name: str) -> str:
    """Generate a reference page for a Provider implementation.

    Mirrors ``_guardrail_page`` but skips guardrail-specific sections
    (SUPPORTED_MODELS, validate) and surfaces the provider's lifecycle
    methods instead (load_model, pre_process, infer, close).
    """
    import importlib

    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)

    lines: list[str] = [f"# {class_name}\n"]

    class_doc = _clean_docstring(inspect.getdoc(cls))
    class_args = _parse_args_section(class_doc)
    if class_doc:
        lines.append(_strip_args_section(class_doc) + "\n")

    init = getattr(cls, "__init__", None)
    if init:
        init_doc = _doc_summary(inspect.getdoc(init))
        lines.append(_section("Constructor"))
        table = _sig_table(init, param_docs={**class_args, **_parse_args_section(inspect.getdoc(init))})
        if table:
            lines.append(table + "\n")
        if init_doc and init_doc != class_doc:
            lines.append(init_doc + "\n")

    for method_name in ("load_model", "pre_process", "infer", "close"):
        method = getattr(cls, method_name, None)
        if method is None:
            continue
        lines.append(_section(method_name))
        method_doc = _doc_summary(inspect.getdoc(method))
        if method_doc:
            lines.append(method_doc + "\n")
        table = _sig_table(method, param_docs=_parse_args_section(inspect.getdoc(method)))
        if table:
            lines.append("**Parameters**\n")
            lines.append(table + "\n")
        ret = _return_annotation(method)
        if ret:
            lines.append(f"**Returns:** `{ret}`\n")

    return "\n".join(lines)


def _any_guardrail_page() -> str:
    from any_guardrail.api import AnyGuardrail

    lines: list[str] = ["# AnyGuardrail\n"]
    class_doc = _clean_docstring(inspect.getdoc(AnyGuardrail))
    if class_doc:
        lines.append(class_doc + "\n")

    for method_name in (
        "create",
        "metadata",
        "list_guardrails",
        "group_by",
        "get_supported_guardrails",
        "get_supported_model",
        "get_all_supported_models",
    ):
        method = getattr(AnyGuardrail, method_name, None)
        if method is None:
            continue
        doc = _doc_summary(inspect.getdoc(method))
        lines.append(_section(method_name))
        if doc:
            lines.append(doc + "\n")
        table = _sig_table(method, param_docs=_parse_args_section(inspect.getdoc(method)))
        if table:
            lines.append("**Parameters**\n")
            lines.append(table + "\n")
        ret = _return_annotation(method)
        if ret:
            lines.append(f"**Returns:** `{ret}`\n")

    return "\n".join(lines)


def _types_page() -> str:
    import any_guardrail.types as types_mod

    lines: list[str] = [
        "# Types\n",
        "Runtime-validated wrappers used throughout the pipeline and the output type returned by every guardrail.\n",
        "A machine-readable JSON Schema for `GuardrailOutput` (generated from these models) is published at "
        "<https://raw.githubusercontent.com/mozilla-ai/any-guardrail/main/schemas/guardrail_output.schema.json>. "
        "Pin a release tag in the URL for a specific version.\n",
    ]

    for cls_name in (
        "GuardrailOutput",
        "CategoryResult",
        "SpanResult",
        "GuardrailUsage",
        "GuardrailPreprocessOutput",
        "GuardrailInferenceOutput",
    ):
        cls = getattr(types_mod, cls_name)
        lines.append(_section(cls_name))
        doc = _clean_docstring(inspect.getdoc(cls))
        if doc:
            lines.append(doc + "\n")
        if hasattr(cls, "model_fields"):
            rows = []
            for field_name, field_info in cls.model_fields.items():
                ann = field_info.annotation
                ann_str = _format_annotation(ann) if ann else "—"
                field_doc = field_info.description or ""
                rows.append(f"| `{field_name}` | `{ann_str}` | {field_doc} |")
            if rows:
                lines.append(
                    "| Field | Type | Description |\n|-------|------|-------------|\n" + "\n".join(rows) + "\n"
                )

    return "\n".join(lines)


def _enum_member_docs(source: str) -> dict[str, list[tuple[str, str]]]:
    """Map each enum class in ``source`` to its ordered ``(value, description)`` members.

    StrEnum member docstrings are attribute docstrings (a string literal following the
    assignment), not present on the member's ``__doc__``, so we read them from source.
    """
    result: dict[str, list[tuple[str, str]]] = {}
    tree = ast.parse(source)
    for node in tree.body:
        if not (isinstance(node, ast.ClassDef) and any(_is_str_enum_base(b) for b in node.bases)):
            continue
        members: list[tuple[str, str]] = []
        body = node.body
        for i, stmt in enumerate(body):
            # `NAME = "value"` optionally followed by a docstring expression.
            if (
                isinstance(stmt, ast.Assign)
                and isinstance(stmt.value, ast.Constant)
                and isinstance(stmt.value.value, str)
            ):
                value = stmt.value.value
                doc = ""
                nxt = body[i + 1] if i + 1 < len(body) else None
                if (
                    isinstance(nxt, ast.Expr)
                    and isinstance(nxt.value, ast.Constant)
                    and isinstance(nxt.value.value, str)
                ):
                    doc = " ".join(nxt.value.value.split())
                members.append((value, doc))
        if members:
            result[node.name] = members
    return result


def _is_str_enum_base(base: Any) -> bool:
    return isinstance(base, ast.Name) and base.id == "StrEnum"


def _taxonomy_page() -> str:
    """Render a human-readable dictionary of the guardrail taxonomy enums.

    Values and their meanings come from ``taxonomy.py`` (the enum member docstrings),
    so this page stays in sync with the taxonomy automatically.
    """
    import any_guardrail.taxonomy as taxonomy_mod

    source = Path(taxonomy_mod.__file__).read_text(encoding="utf-8")
    member_docs = _enum_member_docs(source)

    lines: list[str] = [
        "# Taxonomy\n",
        "The vocabulary behind guardrail metadata (see the [AnyGuardrail reference](any_guardrail.md) "
        "for the `list_guardrails` / `group_by` query API and "
        "[Guardrails](guardrails/index.md) for the catalog grouped by primary category).\n",
        "A machine-readable export of every guardrail's metadata is published at "
        "<https://raw.githubusercontent.com/mozilla-ai/any-guardrail/main/schemas/guardrail_metadata.json>.\n",
    ]

    for enum_name in ("GuardrailCategory", "GuardrailStage", "OutputShape", "BackendType"):
        cls = getattr(taxonomy_mod, enum_name)
        lines.append(_section(enum_name))
        doc = _clean_docstring(inspect.getdoc(cls))
        if doc:
            lines.append(doc + "\n")
        rows = [f"| `{value}` | {_table_cell(desc)} |" for value, desc in member_docs.get(enum_name, [])]
        if rows:
            lines.append("| Value | Meaning |\n|-------|---------|\n" + "\n".join(rows) + "\n")

    return "\n".join(lines)


_CATEGORY_TITLES = {
    "prompt_injection": "Prompt Injection",
    "content_safety": "Content Safety",
    "toxicity": "Toxicity",
    "pii": "PII",
    "hallucination": "Hallucination",
    "off_topic": "Off-Topic",
    "bias": "Bias",
    "tool_use": "Tool Use",
    "general_judge": "General Judge",
}


def _guardrails_index_page() -> str:
    """Render the guardrail index grouped by primary category with flavor text.

    Both the grouping and the descriptions come from the import-free metadata
    registry, so this page stays in sync with the taxonomy automatically.
    """
    from any_guardrail.base import GuardrailName
    from any_guardrail.registry import GUARDRAIL_METADATA
    from any_guardrail.taxonomy import GuardrailCategory

    lines: list[str] = [
        "# Guardrails\n",
        "Available guardrails, grouped by primary category. Select a guardrail to view its API details. "
        "See the [Taxonomy reference](../taxonomy.md) for what each category means.\n",
        "Query this catalog programmatically with `AnyGuardrail.list_guardrails(...)` and "
        "`AnyGuardrail.group_by(...)` — see the [AnyGuardrail reference](../any_guardrail.md).\n",
    ]
    by_primary: dict[GuardrailCategory, list[GuardrailName]] = {}
    for name in GuardrailName:
        by_primary.setdefault(GUARDRAIL_METADATA[name].primary_category, []).append(name)

    for category in GuardrailCategory:
        names = by_primary.get(category, [])
        if not names:
            continue
        lines.append(_section(_CATEGORY_TITLES.get(category.value, category.value.title())))
        lines.append("| Guardrail | Description |")
        lines.append("|-----------|-------------|")
        for name in sorted(names, key=lambda n: GUARDRAIL_METADATA[n].display_name.lower()):
            meta = GUARDRAIL_METADATA[name]
            page = f"{name.value.replace('_', '-')}.md"
            lines.append(f"| [{meta.display_name}]({page}) | {_table_cell(meta.description)} |")
        lines.append("")
    return "\n".join(lines)


def _pascal_class_name(value: str) -> str:
    """PascalCase a GuardrailName value the same way the factory resolves classes."""
    return "".join(part.capitalize() for part in re.split(r"[^A-Za-z0-9]+", value) if part)


# ---------------------------------------------------------------------------
# Guardrail registry: (module_path, ClassName, output_filename)
# ---------------------------------------------------------------------------

PROVIDERS = [
    ("any_guardrail.providers.encoderfile", "EncoderfileProvider", "encoderfile.md"),
    ("any_guardrail.providers.llamafile", "LlamafileProvider", "llamafile.md"),
]


def _guardrail_registry() -> list[tuple[str, str, str]]:
    """Derive (module_path, ClassName, filename) for every guardrail from ``GuardrailName``.

    The naming convention matches ``AnyGuardrail._get_guardrail_class`` (snake_case
    value → module + PascalCase class) and the docs filename convention (dashes), so
    this list never has to be hand-maintained as guardrails are added.
    """
    from any_guardrail.base import GuardrailName

    return [
        (
            f"any_guardrail.guardrails.{name.value}.{name.value}",
            _pascal_class_name(name.value),
            f"{name.value.replace('_', '-')}.md",
        )
        for name in GuardrailName
    ]


GUARDRAILS = _guardrail_registry()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(out_dir: Path | None = None) -> None:
    """Generate API reference Markdown under ``out_dir`` (default ``docs/api``)."""
    api_dir = out_dir or DEFAULT_OUT
    api_dir.mkdir(parents=True, exist_ok=True)
    guardrails_dir = api_dir / "guardrails"
    guardrails_dir.mkdir(parents=True, exist_ok=True)

    path = api_dir / "any_guardrail.md"
    path.write_text(_any_guardrail_page(), encoding="utf-8")
    print(f"  {path}")

    path = api_dir / "types.md"
    path.write_text(_types_page(), encoding="utf-8")
    print(f"  {path}")

    path = api_dir / "taxonomy.md"
    path.write_text(_taxonomy_page(), encoding="utf-8")
    print(f"  {path}")

    path = guardrails_dir / "index.md"
    path.write_text(_guardrails_index_page(), encoding="utf-8")
    print(f"  {path}")

    for module_path, class_name, filename in GUARDRAILS:
        try:
            content = _guardrail_page(module_path, class_name)
        except Exception as exc:
            print(f"  WARNING: falling back to stub for {filename}: {exc}", file=sys.stderr)
            content = _stub_page(class_name)
        path = guardrails_dir / filename
        path.write_text(content, encoding="utf-8")
        print(f"  {path}")

    providers_dir = api_dir / "providers"
    providers_dir.mkdir(parents=True, exist_ok=True)
    for module_path, class_name, filename in PROVIDERS:
        try:
            content = _provider_page(module_path, class_name)
        except Exception as exc:
            print(f"  WARNING: falling back to stub for {filename}: {exc}", file=sys.stderr)
            content = _stub_page(class_name)
        path = providers_dir / filename
        path.write_text(content, encoding="utf-8")
        print(f"  {path}")

    print(f"API docs written to {api_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None, help="Output directory (default: docs/api/)")
    args = parser.parse_args()
    main(args.out)
