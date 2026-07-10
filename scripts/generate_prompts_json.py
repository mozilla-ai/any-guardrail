"""Generate the JSON export of the guardrail prompt registry.

Emits a self-contained JSON document mapping each ``GuardrailName`` value to its
:class:`any_guardrail.prompts.PromptSpec` (its named prompt versions, the default
version, and each template's segments, placeholders, assembly, provenance, and source
URL). The document is committed at ``schemas/guardrail_prompts.json`` so external
tooling — notably the benchmark harness (issue #194) — can pin exactly which prompt
produced a score without importing the package or any model backend.

Only prompt-bearing guardrails appear (the ``PROMPT_REGISTRY`` keys); inline prompt
overrides passed to a guardrail at construction/call time are not stored and never exported.

Dict keys are sorted so the on-disk form is deterministic and ``--check`` diffs are
stable. This script is wired into pre-commit, so CI regenerates the file and fails if
the committed artifact has drifted from the registry.

Usage:
    python scripts/generate_prompts_json.py            # write schemas/guardrail_prompts.json
    python scripts/generate_prompts_json.py --check    # exit non-zero if the committed file is stale
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Ensure the package is importable when run directly.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from any_guardrail.prompt_registry import PROMPT_REGISTRY

DEFAULT_OUT = Path(__file__).parent.parent / "schemas" / "guardrail_prompts.json"


def build_payload() -> dict[str, Any]:
    """Build the ``{guardrail_name: prompt_spec}`` mapping in JSON-native form."""
    return {name.value: spec.model_dump(mode="json") for name, spec in PROMPT_REGISTRY.items()}


def render(payload: dict[str, Any]) -> str:
    """Render the payload to the canonical on-disk string (stable + newline-terminated)."""
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def main() -> int:
    """Generate or check the committed guardrail prompts JSON."""
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
                f"{args.out} is out of date. Run `python scripts/generate_prompts_json.py` and commit the result.",
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
