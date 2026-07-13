"""Generate the JSON export of the guardrail authored-content registry.

Emits a self-contained JSON document mapping each ``GuardrailName`` value that has authored content
to its list of policies / rubrics / criteria (with each item's kind, key, provenance, and source
URL). Committed at ``schemas/guardrail_content.json`` so external tooling can browse the
author-published policies/rubrics/criteria without importing the package or any model backend.

Dict keys and content lists are sorted so the on-disk form is deterministic and ``--check`` diffs
are stable. Wired into pre-commit, so CI regenerates the file and fails on drift from the registry.

Usage:
    python scripts/generate_content_json.py            # write schemas/guardrail_content.json
    python scripts/generate_content_json.py --check    # exit non-zero if the committed file is stale
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from any_guardrail.content_registry import CONTENT_REGISTRY

DEFAULT_OUT = Path(__file__).parent.parent / "schemas" / "guardrail_content.json"


def build_payload() -> dict[str, Any]:
    """Build the ``{guardrail_name: [content, ...]}`` mapping in JSON-native form (sorted by key)."""
    return {
        name.value: [c.model_dump(mode="json") for c in sorted(items, key=lambda c: (c.kind.value, c.key))]
        for name, items in CONTENT_REGISTRY.items()
    }


def render(payload: dict[str, Any]) -> str:
    """Render the payload to the canonical on-disk string (stable + newline-terminated)."""
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def main() -> int:
    """Generate or check the committed guardrail content JSON."""
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
                f"{args.out} is out of date. Run `python scripts/generate_content_json.py` and commit the result.",
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
