import pytest
from any_guardrail.api import AnyGuardrail


def test_create_guardrail_with_invalid_id_raises_error() -> None:
    with pytest.raises(ValueError, match="You tried to instantiate invalid_id"):
        AnyGuardrail.create("invalid_id")
