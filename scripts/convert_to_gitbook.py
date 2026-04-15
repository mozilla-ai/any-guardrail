"""Build the GitBook site output into site/.

Generators write directly into site/ so that docs/api/guardrails/ (which
contains mkdocstrings stubs) never enters the GitBook output.

The contents of site/ are pushed to the gitbook-docs branch by CI.

Usage:
    python scripts/convert_to_gitbook.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DOCS_SRC = Path("docs")
SITE_DIR = Path("site")

SUMMARY = """\
# Table of Contents

* [Introduction](index.md)
* [Quick Start](quickstart.md)

## Cookbook

* [Alinia Guardrail Usage](cookbook/alinia_guardrail_usage.md)
* [Any LLM as a Guardrail](cookbook/any_llm_as_a_guardrail.md)
* [Customer Service Policy Guardrail](cookbook/customer_service_policy_guardrail.md)
* [Custom Blocklists with Azure Content Safety](cookbook/azure_blocklist_slang_filter.md)

## API Reference

* [AnyGuardrail](api/any_guardrail.md)
* [Types](api/types.md)
* [Guardrails](api/guardrails/index.md)
  * [Alinia](api/guardrails/alinia.md)
  * [AnyLLM](api/guardrails/any-llm.md)
  * [Azure Content Safety](api/guardrails/azure-content-safety.md)
  * [Deepset](api/guardrails/deepset.md)
  * [DuoGuard](api/guardrails/duo-guard.md)
  * [FlowJudge](api/guardrails/flowjudge.md)
  * [Glider](api/guardrails/glider.md)
  * [HarmGuard](api/guardrails/harm-guard.md)
  * [InjecGuard](api/guardrails/injec-guard.md)
  * [Jasper](api/guardrails/jasper.md)
  * [LlamaGuard](api/guardrails/llama-guard.md)
  * [OffTopic](api/guardrails/off-topic.md)
  * [Pangolin](api/guardrails/pangolin.md)
  * [ProtectAI](api/guardrails/protectai.md)
  * [Sentinel](api/guardrails/sentinel.md)
  * [ShieldGemma](api/guardrails/shield-gemma.md)
"""

# Plain Markdown docs to copy verbatim from docs/ (excludes api/guardrails/ — generated directly)
STATIC_MD_GLOBS = ["*.md", "api/*.md"]

# Static asset directories
ASSET_DIRS = ["images", "assets"]


def copy_static_docs() -> None:
    """Copy top-level and api-level Markdown from docs/ into site/."""
    for glob in STATIC_MD_GLOBS:
        for src in sorted(DOCS_SRC.glob(glob)):
            rel = src.relative_to(DOCS_SRC)
            dst = SITE_DIR / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def copy_assets() -> None:
    """Copy static asset directories from docs/ into site/."""
    for subdir in ASSET_DIRS:
        src = DOCS_SRC / subdir
        if src.exists():
            shutil.copytree(src, SITE_DIR / subdir)
            print(f"  Copied {len(list(src.rglob('*')))} assets from docs/{subdir}/")


def validate_summary(summary: str, site_dir: Path) -> None:
    """Assert every page listed in SUMMARY.md exists under site/."""
    missing = []
    for line in summary.splitlines():
        if "(" not in line or ")" not in line:
            continue
        start = line.index("(") + 1
        end = line.index(")", start)
        path = line[start:end].strip()
        if not path or path.startswith("http"):
            continue
        if not (site_dir / path).exists():
            missing.append(path)

    if missing:
        print("ERROR: the following SUMMARY.md pages are missing from site/:", file=sys.stderr)
        for p in missing:
            print(f"  {p}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    """Build the GitBook-flavored Markdown site into ``site/``."""
    # Create site/ fresh
    if SITE_DIR.exists():
        shutil.rmtree(SITE_DIR)
    SITE_DIR.mkdir()

    # Generate API docs directly into site/api/ (bypasses docs/api/guardrails/ stubs)
    print("Generating API docs...")
    sys.path.insert(0, str(SCRIPT_DIR))
    import generate_api_docs

    generate_api_docs.main(SITE_DIR / "api")

    # Generate cookbooks directly into site/cookbook/
    print("Converting cookbooks...")
    import generate_cookbooks

    generate_cookbooks.main(SITE_DIR / "cookbook")

    # Copy plain Markdown docs (index.md, quickstart.md, api/*.md top-level)
    print("Copying static docs...")
    copy_static_docs()
    copy_assets()

    # Write navigation and GitBook config
    (SITE_DIR / "SUMMARY.md").write_text(SUMMARY, encoding="utf-8")
    gitbook_yaml = Path(".gitbook.yaml")
    if gitbook_yaml.exists():
        shutil.copy2(gitbook_yaml, SITE_DIR / ".gitbook.yaml")

    # Validate before finishing
    print("Validating SUMMARY.md...")
    validate_summary(SUMMARY, SITE_DIR)

    md_count = len(list(SITE_DIR.rglob("*.md")))
    print(f"\nDone — {md_count} Markdown files written to {SITE_DIR}/")


if __name__ == "__main__":
    main()
