"""Regenerate ``src/any_guardrail/_authored_content_data.py`` from live sources.

Mirrors the Granite Guardian risk criteria (from ``GraniteGuardianRisk``) and the Flow-Judge
preset criteria + rubrics (from the installed ``flow_judge`` library) into a stdlib-only data
module so the import-free content registry can expose them without importing those backends.

Values are written as ``json.dumps`` string literals (double-quoted, escaped) so the output is
deterministic and ruff-clean. Requires the ``huggingface`` and ``flowjudge`` extras. This is NOT a
pre-commit hook (it imports heavy backends); byte-exactness is enforced instead by the drift tests
in ``tests/unit/test_content.py``.

Usage:
    python scripts/generate_content_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

OUT = Path(__file__).parent.parent / "src" / "any_guardrail" / "_authored_content_data.py"

_HEADER = '''"""Generated verbatim authored content mirrored for the import-free content registry.

Granite Guardian risk criteria come from ``GraniteGuardianRisk``; Flow-Judge metric criteria and
rubrics come from the installed ``flow_judge`` presets. Kept byte-identical to those sources by the
drift tests in ``tests/unit/test_content.py`` (the Flow-Judge checks skip without the flowjudge
extra). Regenerate with ``scripts/generate_content_data.py``."""

'''


def _dict_src(name: str, mapping: dict[str, str]) -> str:
    lines = [f"{name} = {{"]
    lines += [f"    {json.dumps(k)}: {json.dumps(v)}," for k, v in mapping.items()]
    lines.append("}")
    return "\n".join(lines)


def main() -> int:
    """Write the data module from the live Granite / Flow-Judge sources."""
    import flow_judge.metrics as metrics
    from flow_judge.utils.prompt_formatter import format_rubric

    from any_guardrail.guardrails.granite_guardian.granite_guardian import GraniteGuardianRisk

    granite = {a.lower(): getattr(GraniteGuardianRisk, a) for a in vars(GraniteGuardianRisk) if a.isupper()}
    presets = [n for n in dir(metrics) if n.isupper() and not n.startswith("_")]
    fj_criteria = {n.lower(): getattr(metrics, n).criteria for n in presets}
    fj_rubrics = {n.lower(): format_rubric(getattr(metrics, n).rubric) for n in presets}

    body = "\n\n".join(
        [
            _dict_src("GRANITE_CRITERIA", granite),
            _dict_src("FLOWJUDGE_CRITERIA", fj_criteria),
            _dict_src("FLOWJUDGE_RUBRICS", fj_rubrics),
        ]
    )
    OUT.write_text(_HEADER + body + "\n", encoding="utf-8")
    print(f"Wrote {OUT} ({len(granite)} granite criteria, {len(fj_criteria)} flow-judge presets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
