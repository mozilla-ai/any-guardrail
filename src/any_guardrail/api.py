from any_guardrail.utils.model_registry import model_registry
from any_guardrail.guardrails.guardrail import Guardrail
from typing import List, Any


def list_all_supported_guardrails() -> List[str]:
    return list(model_registry.keys())


def create_guardrail(guardrail_identifier: str, **kwargs: Any) -> Guardrail:
    if guardrail_identifier in model_registry.keys():
        registry = model_registry[guardrail_identifier](model_identifier=guardrail_identifier, **kwargs)
        if not isinstance(registry, Guardrail):
            raise ValueError(
                f"{guardrail_identifier} is not of type Guardrail. Please use the correct model identifier."
            )
        return registry
    else:
        raise ValueError(
            f"You tried to instantiate {guardrail_identifier}, which is not a supported guardrail. "
            f"Use list_all_supported_guardrails to see which guardrails are supported."
        )
