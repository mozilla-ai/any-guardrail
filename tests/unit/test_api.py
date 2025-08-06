import pytest
from any_guardrail.api import GuardrailFactory


def test_create_guardrail_with_invalid_id_raises_error() -> None:
    factory = GuardrailFactory()

    with pytest.raises(ValueError, match="You tried to instantiate invalid_id"):
        factory.create_guardrail("invalid_id")
