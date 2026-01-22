from any_guardrail import AnyGuardrail, GuardrailName, GuardrailOutput


def test_azure_content_safety_guardrail_integration() -> None:
    guardrail = AnyGuardrail.create(GuardrailName.AZURE_CONTENT_SAFETY)

    result: GuardrailOutput[bool, dict[str, int | list[str] | None], float] = guardrail.validate(
        "This is a piece of benign text."
    )

    assert isinstance(result, GuardrailOutput)
    assert result.valid
    assert result.explanation is not None
    assert result.score == 0


def test_azure_content_safety_guardrail_integration_flagged() -> None:
    guardrail = AnyGuardrail.create(GuardrailName.AZURE_CONTENT_SAFETY)

    result: GuardrailOutput[bool, dict[str, int | list[str] | None], float] = guardrail.validate(
        "I want to hurt myself and everyone around violently and brutally until there's nothing left."
    )

    assert isinstance(result, GuardrailOutput)
    assert not result.valid
    assert result.explanation is not None
    assert result.score > 2  # type: ignore [operator]


def test_azure_content_safety_guardrail_integration_blocklist() -> None:
    guardrail = AnyGuardrail.create(
        GuardrailName.AZURE_CONTENT_SAFETY,
        blocklist_names=["Test"],
    )

    guardrail.create_or_update_blocklist(  # type: ignore [attr-defined]
        blocklist_name="Test",
        blocklist_description="Blocklist for testing purposes",
    )

    guardrail.add_blocklist_items(  # type: ignore [attr-defined]
        blocklist_name="Test",
        blocklist_terms=["ham", "sandwich"],
    )

    result: GuardrailOutput[bool, dict[str, int | list[str] | None], float] = guardrail.validate(
        "This is some ham and sandwich content that should be blocked."
    )

    assert isinstance(result, GuardrailOutput)
    assert not result.valid
    assert result.explanation is not None
    assert result.explanation.get("blocklist") is not None

    guardrail.delete_blocklist(blocklist_name="Test")  # type: ignore [attr-defined]
