"""Generate the JSON Schema for a committed benchmark result (#194).

Emits a self-contained JSON Schema (draft 2020-12) for ``BenchmarkResult`` — including its nested
``BenchmarkSource`` and ``ComparisonCohort`` component models — derived directly from the Pydantic
definitions. Committed at ``schemas/guardrail_benchmarks.schema.json`` so the committed benchmark
results (``schemas/guardrail_benchmarks.json``) can be validated against a versioned contract by
external tooling.

Like the GuardrailOutput schema, the committed artifact is structural only: no ``$id`` and no
embedded package version (git tags provide versioning).

Usage:
    python scripts/generate_benchmark_schema.py            # write schemas/guardrail_benchmarks.schema.json
    python scripts/generate_benchmark_schema.py --check    # exit non-zero if the committed file is stale
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from any_guardrail.benchmarks import BenchmarkResult

DEFAULT_OUT = Path(__file__).parent.parent / "schemas" / "guardrail_benchmarks.schema.json"


def _strip_redundant_keywords(node: Any) -> None:
    """Drop JSON-Schema keywords that restate the default (e.g. ``additionalProperties: true``)."""
    if isinstance(node, dict):
        if node.get("additionalProperties") is True:
            del node["additionalProperties"]
        for value in node.values():
            _strip_redundant_keywords(value)
    elif isinstance(node, list):
        for item in node:
            _strip_redundant_keywords(item)


def build_schema() -> dict[str, Any]:
    """Build the JSON Schema dict for ``BenchmarkResult`` (draft 2020-12, no ``$id``)."""
    schema = BenchmarkResult.model_json_schema()
    _strip_redundant_keywords(schema)
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", **schema}


def render(schema: dict[str, Any]) -> str:
    """Render the schema to the canonical on-disk string (stable + newline-terminated)."""
    return json.dumps(schema, indent=2, sort_keys=True) + "\n"


def main() -> int:
    """Generate or check the committed BenchmarkResult JSON Schema."""
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
                f"{args.out} is out of date. Run `python scripts/generate_benchmark_schema.py` and commit the result.",
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
