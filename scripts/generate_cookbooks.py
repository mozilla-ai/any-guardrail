"""Convert Jupyter notebooks in docs/cookbook/ to GitBook-compatible Markdown.

Reads each .ipynb file, converts cells to Markdown/code blocks, and writes
the result to a given output directory. Generated .md files should never
be committed to git.

Usage:
    python scripts/generate_cookbooks.py                         # writes to docs/cookbook/
    python scripts/generate_cookbooks.py --out site/cookbook     # writes to site/cookbook/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
COOKBOOKS_SRC = REPO_ROOT / "docs" / "cookbook"
GITHUB_COLAB_BASE = (
    "https://colab.research.google.com/github/mozilla-ai/any-guardrail/blob/main/docs/cookbook"
)


def _is_install_only_cell(source: str) -> bool:
    """Return True for cells that contain only pip installs or magic commands."""
    lines = [ln.strip() for ln in source.strip().splitlines() if ln.strip()]
    return bool(lines) and all(ln.startswith(("%", "!", "#")) for ln in lines)


def _strip_magic_lines(source: str) -> str:
    """Remove IPython magic and shell lines from a code cell."""
    kept = [ln for ln in source.splitlines() if not ln.lstrip().startswith(("%", "!"))]
    return "\n".join(kept).strip()


def notebook_to_md(notebook_path: Path) -> str:
    """Convert a single .ipynb file to a Markdown string."""
    with open(notebook_path, encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb.get("cells", [])
    if not cells:
        return ""

    kernel_lang = nb.get("metadata", {}).get("kernelspec", {}).get("language", "python")

    # Extract title from first markdown heading
    title = notebook_path.stem.replace("_", " ").title()
    first_source = "".join(cells[0].get("source", []))
    if first_source.strip().startswith("# "):
        title = first_source.strip().lstrip("# ").splitlines()[0].strip()

    colab_url = f"{GITHUB_COLAB_BASE}/{notebook_path.name}"

    lines: list[str] = []
    lines.append(f"# {title}\n")
    lines.append(
        f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab_url})\n"
    )

    first_heading_skipped = False

    for cell in cells:
        source = "".join(cell.get("source", []))
        if not source.strip():
            continue

        cell_type = cell.get("cell_type", "code")

        if cell_type == "markdown":
            stripped = source.strip()
            # Skip the title heading — already written above
            if not first_heading_skipped and stripped.startswith("# "):
                first_heading_skipped = True
                continue
            # Skip standalone Colab badge cells
            if "colab-badge.svg" in stripped and len(stripped.splitlines()) <= 3:
                continue
            lines.append(source.rstrip())
            lines.append("")

        elif cell_type == "code":
            if _is_install_only_cell(source):
                continue
            cleaned = _strip_magic_lines(source)
            if not cleaned:
                continue
            lines.append(f"```{kernel_lang}")
            lines.append(cleaned)
            lines.append("```")
            lines.append("")

    return "\n".join(lines)


def main(out_dir: Path | None = None) -> None:
    dest = out_dir or COOKBOOKS_SRC
    dest.mkdir(parents=True, exist_ok=True)

    notebooks = sorted(COOKBOOKS_SRC.glob("*.ipynb"))
    if not notebooks:
        print(f"No notebooks found in {COOKBOOKS_SRC}", file=sys.stderr)
        return

    for nb_path in notebooks:
        md_content = notebook_to_md(nb_path)
        out_path = dest / nb_path.with_suffix(".md").name
        out_path.write_text(md_content, encoding="utf-8")
        print(f"  {out_path}")

    print(f"Cookbooks written ({len(notebooks)} notebooks converted).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None, help="Output directory (default: docs/cookbook/)")
    args = parser.parse_args()
    main(args.out)
