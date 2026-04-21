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


def _format_default(default: Any) -> str:
    if default is inspect.Parameter.empty:
        return ""
    if default is None:
        return "None"
    if isinstance(default, str):
        return f'"{default}"'
    return repr(default)


def _sig_table(func: Any, skip_self: bool = True) -> str:
    try:
        sig = inspect.signature(func)
    except (ValueError, TypeError):
        return ""

    rows = []
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
        rows.append(f"| `{name}` | {ann_cell} | {required} | {default_cell} |")

    if not rows:
        return ""
    header = "| Parameter | Type | Required | Default |\n|-----------|------|----------|---------|\n"
    return header + "\n".join(rows)


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


def _section(title: str, level: int = 2) -> str:
    return f"{'#' * level} {title}\n"


# ---------------------------------------------------------------------------
# Page generators
# ---------------------------------------------------------------------------


def _guardrail_page(module_path: str, class_name: str) -> str:
    import importlib

    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)

    lines: list[str] = [f"# {class_name}\n"]

    class_doc = _clean_docstring(inspect.getdoc(cls))
    if class_doc:
        lines.append(class_doc + "\n")

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
        table = _sig_table(init)
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
        table = _sig_table(validate)
        if table:
            lines.append("**Parameters**\n")
            lines.append(table + "\n")
        ret = _return_annotation(validate)
        if ret:
            lines.append(f"**Returns:** `{ret}`\n")

    return "\n".join(lines)


def _stub_page(class_name: str) -> str:
    """Minimal page for a guardrail whose heavy deps aren't installed."""
    return (
        f"# {class_name}\n\n"
        "> API reference for this guardrail requires its optional dependencies to be installed.\n\n"
        f"Install with: `pip install 'any-guardrail[{class_name.lower()}]'`\n"
    )


def _any_guardrail_page() -> str:
    from any_guardrail.api import AnyGuardrail

    lines: list[str] = ["# AnyGuardrail\n"]
    class_doc = _clean_docstring(inspect.getdoc(AnyGuardrail))
    if class_doc:
        lines.append(class_doc + "\n")

    for method_name in ("create", "get_supported_guardrails", "get_supported_model", "get_all_supported_models"):
        method = getattr(AnyGuardrail, method_name, None)
        if method is None:
            continue
        doc = _doc_summary(inspect.getdoc(method))
        lines.append(_section(method_name))
        if doc:
            lines.append(doc + "\n")
        table = _sig_table(method)
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
    ]

    for cls_name in ("GuardrailOutput", "GuardrailPreprocessOutput", "GuardrailInferenceOutput"):
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


def _guardrails_index_page() -> str:
    from any_guardrail.base import GuardrailName

    lines: list[str] = [
        "# Guardrails\n",
        "Available guardrails and their parameters. Select a guardrail to view its API details.\n",
        "| Name | `GuardrailName` value |",
        "|------|-----------------------|",
    ]
    for name in GuardrailName:
        title = name.name.capitalize()
        page = name.value.replace("_", "-")
        lines.append(f"| [{title}]({page}.md) | `GuardrailName.{name.name}` |")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Guardrail registry: (module_path, ClassName, output_filename)
# ---------------------------------------------------------------------------

GUARDRAILS = [
    ("any_guardrail.guardrails.alinia.alinia", "Alinia", "alinia.md"),
    ("any_guardrail.guardrails.any_llm.any_llm", "AnyLlm", "any-llm.md"),
    (
        "any_guardrail.guardrails.azure_content_safety.azure_content_safety",
        "AzureContentSafety",
        "azure-content-safety.md",
    ),
    ("any_guardrail.guardrails.deepset.deepset", "Deepset", "deepset.md"),
    ("any_guardrail.guardrails.duo_guard.duo_guard", "DuoGuard", "duo-guard.md"),
    ("any_guardrail.guardrails.flowjudge.flowjudge", "Flowjudge", "flowjudge.md"),
    ("any_guardrail.guardrails.glider.glider", "Glider", "glider.md"),
    ("any_guardrail.guardrails.harm_guard.harm_guard", "HarmGuard", "harm-guard.md"),
    ("any_guardrail.guardrails.injec_guard.injec_guard", "InjecGuard", "injec-guard.md"),
    ("any_guardrail.guardrails.jasper.jasper", "Jasper", "jasper.md"),
    ("any_guardrail.guardrails.llama_guard.llama_guard", "LlamaGuard", "llama-guard.md"),
    ("any_guardrail.guardrails.off_topic.off_topic", "OffTopic", "off-topic.md"),
    ("any_guardrail.guardrails.pangolin.pangolin", "Pangolin", "pangolin.md"),
    ("any_guardrail.guardrails.protectai.protectai", "Protectai", "protectai.md"),
    ("any_guardrail.guardrails.sentinel.sentinel", "Sentinel", "sentinel.md"),
    ("any_guardrail.guardrails.shield_gemma.shield_gemma", "ShieldGemma", "shield-gemma.md"),
]


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

    print(f"API docs written to {api_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None, help="Output directory (default: docs/api/)")
    args = parser.parse_args()
    main(args.out)
