"""Generate the JSON export of the guardrail taxonomy metadata registry.

Emits a self-contained JSON document mapping each ``GuardrailName`` value to its
:class:`any_guardrail.taxonomy.GuardrailMetadata` (categories, stages, output
shapes, backend, validate() kwargs, and the secondary capability flags). The
document is committed at ``schemas/guardrail_metadata.json`` so external tooling
can query the taxonomy — "which guardrails detect prompt injection?" — without
importing the package or any model backend.

Set-valued fields are serialized as sorted lists (via ``GuardrailMetadata``'s
field serializer) and dict keys are sorted, so the on-disk form is deterministic
and ``--check`` diffs are stable.

This script is wired into pre-commit, so CI regenerates the file and fails if the
committed artifact has drifted from the registry.

Usage:
    python scripts/generate_metadata_json.py            # write schemas/guardrail_metadata.json
    python scripts/generate_metadata_json.py --check    # exit non-zero if the committed file is stale
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Ensure the package is importable when run directly.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from any_guardrail.registry import GUARDRAIL_METADATA

DEFAULT_OUT = Path(__file__).parent.parent / "schemas" / "guardrail_metadata.json"


def build_payload() -> dict[str, Any]:
    """Build the ``{guardrail_name: metadata}`` mapping in JSON-native form."""
    return {name.value: meta.model_dump(mode="json") for name, meta in GUARDRAIL_METADATA.items()}


def render(payload: dict[str, Any]) -> str:
    """Render the payload to the canonical on-disk string (stable + newline-terminated)."""
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def main() -> int:
    """Generate or check the committed guardrail metadata JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"Output path (default: {DEFAULT_OUT})")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero (without writing) if the committed file differs from the registry.",
    )
    args = parser.parse_args()

    content = render(build_payload())

    if args.check:
        if not args.out.exists() or args.out.read_text(encoding="utf-8") != content:
            print(
                f"{args.out} is out of date. Run `python scripts/generate_metadata_json.py` and commit the result.",
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
