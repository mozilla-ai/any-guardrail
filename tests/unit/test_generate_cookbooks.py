import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import generate_cookbooks


def test_encoderfile_cookbook_keeps_install_step() -> None:
    cookbook = REPO_ROOT / "docs" / "cookbook" / "encoderfile_guardrail.ipynb"

    rendered = generate_cookbooks.notebook_to_md(cookbook)

    assert "## Install" in rendered
    assert "```bash\npip install 'any-guardrail[encoderfile,huggingface]' --quiet\n```" in rendered


def test_install_only_cells_render_as_bash(tmp_path: Path) -> None:
    notebook = {
        "cells": [
            {"cell_type": "markdown", "source": ["# Sample\n"]},
            {
                "cell_type": "code",
                "source": [
                    "%pip install any-guardrail\n",
                    "# keep comments\n",
                    "!python -m pip install pytest\n",
                ],
            },
        ],
        "metadata": {"kernelspec": {"language": "python"}},
    }
    notebook_path = tmp_path / "sample.ipynb"
    notebook_path.write_text(json.dumps(notebook), encoding="utf-8")

    rendered = generate_cookbooks.notebook_to_md(notebook_path)

    assert "```bash\npip install any-guardrail\n# keep comments\npython -m pip install pytest\n```" in rendered
