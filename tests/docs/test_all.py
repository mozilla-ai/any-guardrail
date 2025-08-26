import pathlib
from unittest.mock import MagicMock, patch

import pytest
from mktestdocs import check_md_file


@pytest.mark.parametrize(
    "doc_file",
    list(pathlib.Path("docs").glob("**/*.md")),
    ids=str,
)
def test_all_docs(doc_file: pathlib.Path) -> None:
    mocked_guardrail_class = MagicMock()
    mocked_guardrail_output = MagicMock()
    mocked_guardrail_output.valid = True
    mocked_guardrail_class.validate.return_value = mocked_guardrail_output

    mocked_get_guardrail_class = MagicMock()
    mocked_get_guardrail_class.return_value.return_value = mocked_guardrail_class

    with (
        patch("any_guardrail.api.AnyGuardrail._get_guardrail_class", mocked_get_guardrail_class),
    ):
        check_md_file(fpath=doc_file, memory=True)  # type: ignore[no-untyped-call]
