"""Generate the JSON Schema for the GuardrailOutput standard.

Emits a self-contained JSON Schema (draft 2020-12) for ``GuardrailOutput`` —
including its nested ``CategoryResult``, ``SpanResult`` and ``GuardrailUsage``
component models — derived directly from the Pydantic definitions. The schema
is committed at ``schemas/guardrail_output.schema.json`` so it is browsable on
GitHub and can be referenced from docs and downstream consumers.

The committed artifact is intentionally *structural only*: it does not embed the
package version (which is a volatile setuptools_scm dev string between releases
and would churn on every commit). Versioning is provided by git — the schema at
a given tag is that release's schema — so pin a release tag in the raw URL when
you need a specific version.

This script is wired into pre-commit, so CI regenerates the schema and fails if
the committed file has drifted from the models.

Usage:
    python scripts/generate_json_schema.py            # write schemas/guardrail_output.schema.json
    python scripts/generate_json_schema.py --check    # exit non-zero if the committed file is stale
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Ensure the package is importable when run directly.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from any_guardrail import GuardrailOutput

DEFAULT_OUT = Path(__file__).parent.parent / "schemas" / "guardrail_output.schema.json"
SCHEMA_ID = "https://raw.githubusercontent.com/mozilla-ai/any-guardrail/main/schemas/guardrail_output.schema.json"


def build_schema() -> dict[str, Any]:
    """Build the enriched JSON Schema dict for ``GuardrailOutput``."""
    schema = GuardrailOutput.model_json_schema()
    # Pydantic v2 emits draft 2020-12; advertise the dialect and a stable $id so
    # the document is a self-describing, referenceable schema.
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": SCHEMA_ID,
        **schema,
    }


def render(schema: dict[str, Any]) -> str:
    """Render the schema to the canonical on-disk string (stable + newline-terminated)."""
    return json.dumps(schema, indent=2, sort_keys=True) + "\n"


def main() -> int:
    """Generate or check the committed GuardrailOutput JSON Schema."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"Output path (default: {DEFAULT_OUT})")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero (without writing) if the committed schema differs from the models.",
    )
    args = parser.parse_args()

    content = render(build_schema())

    if args.check:
        if not args.out.exists() or args.out.read_text(encoding="utf-8") != content:
            print(
                f"{args.out} is out of date. Run `python scripts/generate_json_schema.py` and commit the result.",
                file=sys.stderr,
            )
            return 1
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(content, encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
