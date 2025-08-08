from any_guardrail.utils.model_registry import model_registry
from any_guardrail.guardrails.guardrail import Guardrail
from typing import Any


class AnyGuardrail:
    """Factory class for creating and managing guardrail instances."""

    supported_guardrails = list(model_registry.keys())

    @classmethod
    def create(cls, model_id: str, **kwargs: Any) -> Guardrail:
        """Create a guardrail instance.

        Args:
            model_id: The identifier of the model to use, which will be mapped to the guardrail that uses it.
            **kwargs: Additional keyword arguments to pass to the guardrail constructor.

        Returns:
            A guardrail instance.
        """
        if model_id in model_registry.keys():
            registry = model_registry[model_id](model_id=model_id, **kwargs)
            if not isinstance(registry, Guardrail):
                raise ValueError(f"{model_id} is not of type Guardrail. Please use the correct model identifier.")
            return registry
        else:
            raise ValueError(
                f"You tried to instantiate {model_id}, which is not a supported guardrail. "
                f"Use list_all_supported_guardrails to see which guardrails are supported."
            )
