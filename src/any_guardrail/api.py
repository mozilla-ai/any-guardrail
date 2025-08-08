from any_guardrail.utils.model_registry import model_registry
from any_guardrail.guardrails.guardrail import Guardrail
from typing import List, Any


class GuardrailFactory:
    """Factory class for creating and managing guardrail instances."""

    # This is a class variable so that it can be accessed by the create_guardrail method
    # and the list_all_supported_guardrails method
    model_registry = model_registry

    @classmethod
    def list_all_supported_guardrails(cls) -> List[str]:
        """List all supported guardrails."""
        return list(cls.model_registry.keys())

    @classmethod
    def create_guardrail(cls, model_id: str, **kwargs: Any) -> Guardrail:
        """Create a guardrail instance.

        Args:
            model_id: The identifier of the model to use, which will be mapped to the guardrail that uses it.
            **kwargs: Additional keyword arguments to pass to the guardrail constructor.

        Returns:
            A guardrail instance.
        """
        if model_id in cls.model_registry.keys():
            registry = cls.model_registry[model_id](model_id=model_id, **kwargs)
            if not isinstance(registry, Guardrail):
                raise ValueError(f"{model_id} is not of type Guardrail. Please use the correct model identifier.")
            return registry
        else:
            raise ValueError(
                f"You tried to instantiate {model_id}, which is not a supported guardrail. "
                f"Use list_all_supported_guardrails to see which guardrails are supported."
            )
