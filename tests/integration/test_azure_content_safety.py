from any_guardrail import AnyGuardrail, GuardrailName, GuardrailOutput


def test_azure_content_safety_guardrail_integration() -> None:
    guardrail = AnyGuardrail.create(GuardrailName.AZURE_CONTENT_SAFETY)

    result = guardrail.validate("This is a piece of benign text.")

    assert isinstance(result, GuardrailOutput)
    assert result.valid
    assert result.explanation is not None
    assert result.score == 0


def test_azure_content_safety_guardrail_integration_flagged() -> None:
    guardrail = AnyGuardrail.create(GuardrailName.AZURE_CONTENT_SAFETY)

    result = guardrail.validate(
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
        blocklist_names="Test",
        blocklist_description="Blocklist for testing purposes",
    )

    guardrail.add_blocklist_items(  # type: ignore [attr-defined]
        blocklist_names="Test",
        blocklist_terms=["ham", "sandwich"],
    )

    result = guardrail.validate("This is some ham and sandwich content that should be blocked.")

    assert isinstance(result, GuardrailOutput)
    assert not result.valid
    assert result.explanation is not None
    assert result.explanation.get("blocklist") is not None  # type: ignore [union-attr]

    guardrail.delete_blocklist(blocklist_names="Test")  # type: ignore [attr-defined]
