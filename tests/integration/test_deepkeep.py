import os

import pytest

from any_guardrail import AnyGuardrail, GuardrailName, GuardrailOutput


@pytest.mark.parametrize(
    ("text_variable", "expected_valid"),
    [
        ("DEEPKEEP_SAFE_TEXT", True),
        ("DEEPKEEP_BLOCKED_TEXT", False),
    ],
)
def test_live_firewall(text_variable: str, expected_valid: bool) -> None:
    required = ["DEEPKEEP_API_KEY", "DEEPKEEP_API_BASE", "DEEPKEEP_INPUT_FIREWALL_ID", text_variable]
    if any(not os.getenv(name) for name in required):
        pytest.skip("Set DeepKeep deployment credentials and policy-specific safe/blocked test text to run.")
    guard = AnyGuardrail.create(
        GuardrailName.DEEPKEEP,
        api_key=os.environ["DEEPKEEP_API_KEY"],
        endpoint=os.environ["DEEPKEEP_API_BASE"],
        firewall_id=os.environ["DEEPKEEP_INPUT_FIREWALL_ID"],
    )
    result = AnyGuardrail.evaluate(GuardrailName.DEEPKEEP, guard, os.environ[text_variable])
    assert isinstance(result, GuardrailOutput)
    assert result.valid is expected_valid
