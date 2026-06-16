from any_guardrail import AnyGuardrail, GuardrailName, GuardrailOutput


def test_azure_content_safety_guardrail_integration() -> None:
    guardrail = AnyGuardrail.create(GuardrailName.AZURE_CONTENT_SAFETY)

    result: GuardrailOutput = guardrail.validate("This is a piece of benign text.")

    assert isinstance(result, GuardrailOutput)
    assert result.valid
    assert {category.name for category in result.categories} == {"hate", "self_harm", "sexual", "violence"}
    assert result.score == 0.0


def test_azure_content_safety_guardrail_integration_flagged() -> None:
    guardrail = AnyGuardrail.create(GuardrailName.AZURE_CONTENT_SAFETY)

    result: GuardrailOutput = guardrail.validate(
        "I want to hurt myself and everyone around violently and brutally until there's nothing left."
    )

    assert isinstance(result, GuardrailOutput)
    assert not result.valid
    assert any(category.triggered for category in result.categories)
    assert result.score > 2 / 7  # type: ignore [operator]


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

    result: GuardrailOutput = guardrail.validate("This is some ham and sandwich content that should be blocked.")

    assert isinstance(result, GuardrailOutput)
    assert not result.valid
    assert result.extra is not None
    assert result.extra["blocklists_match"]

    guardrail.delete_blocklist(blocklist_name="Test")  # type: ignore [attr-defined]
