"""Auto-mark every test under ``tests/integration/`` as ``e2e``.

By convention this directory holds tests that exercise real binaries, real
models downloaded from HuggingFace, or real external APIs — i.e. e2e tests.
Rather than annotate each module with ``pytestmark = pytest.mark.e2e``,
we apply the marker once here so new integration tests pick it up
automatically. Individual files only need to add ``pytest.mark.heavy``
(or other narrower marks) when they apply.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterable


def pytest_collection_modifyitems(items: Iterable[pytest.Item]) -> None:
    """Add the ``e2e`` marker to every test collected from this directory."""
    e2e = pytest.mark.e2e
    for item in items:
        # ``pytest_collection_modifyitems`` is called with the global item list
        # even when the conftest lives in a subdirectory. Scope the marker
        # application to items whose nodeid lives under this directory.
        if "tests/integration/" in item.nodeid.replace("\\", "/"):
            item.add_marker(e2e)
