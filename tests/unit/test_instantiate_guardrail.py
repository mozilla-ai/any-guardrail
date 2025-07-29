import pytest
from unittest.mock import patch, MagicMock
from any_guardrail import call_guardrail


def test_list_guardrail_classes():
    with (
        patch("any_guardrail.instantiate_guardrail._discover_guardrail_classes", return_value=None),
        patch.dict("any_guardrail.instantiate_guardrail._GUARDRAIL_CLASSES", {"jasper": MagicMock()}),
    ):
        classes = call_guardrail.list_guardrail_classes()
        assert "jasper" in classes


def test_instantiate_guardrail_success():
    Dummy = type("Dummy", (), {"__init__": lambda self, x=1: None})
    with (
        patch("any_guardrail.instantiate_guardrail._discover_guardrail_classes", return_value=None),
        patch.dict("any_guardrail.instantiate_guardrail._GUARDRAIL_CLASSES", {"dummy": Dummy}),
    ):
        obj = call_guardrail.instantiate_guardrail("dummy", x=2)
        assert isinstance(obj, Dummy)


def test_instantiate_guardrail_not_found():
    with (
        patch("any_guardrail.instantiate_guardrail._discover_guardrail_classes", return_value=None),
        patch.dict("any_guardrail.instantiate_guardrail._GUARDRAIL_CLASSES", {}),
    ):
        with pytest.raises(ValueError):
            call_guardrail.instantiate_guardrail("notaclass")


def test_instantiate_guardrail_type_error():
    Dummy = type("Dummy", (), {"__init__": lambda self, x: None})
    with (
        patch("any_guardrail.instantiate_guardrail._discover_guardrail_classes", return_value=None),
        patch.dict("any_guardrail.instantiate_guardrail._GUARDRAIL_CLASSES", {"dummy": Dummy}),
    ):
        with pytest.raises(ValueError):
            call_guardrail.instantiate_guardrail("dummy")
