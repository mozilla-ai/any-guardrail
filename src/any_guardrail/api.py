import importlib
from typing import List, Any

from any_guardrail.guardrail import Guardrail, GuardrailName

import inspect
import re


class GuardrailFactory:
    """Factory class for creating guardrail instances."""

    @classmethod
    def get_supported_guardrails(cls) -> List[GuardrailName]:
        """List all supported guardrails."""
        return list(GuardrailName)

    @classmethod
    def create_guardrail(cls, guardrail_name: GuardrailName, model_id: str, **kwargs: Any) -> Guardrail:
        """Create a guardrail instance.

        Args:
            guardrail_name: The name of the guardrail to use.
            model_id: The identifier of the model to use, which will be mapped to the guardrail that uses it.
            **kwargs: Additional keyword arguments to pass to the guardrail constructor.

        Returns:
            A guardrail instance.
        """
        guardrail_module_name = f"{guardrail_name.value}"
        module_path = f"any_guardrail.guardrails.{guardrail_module_name}"

        module = importlib.import_module(module_path)
        parts = re.split(r"[^A-Za-z0-9]+", guardrail_module_name)
        candidate_name = "".join(p.capitalize() for p in parts if p)
        guardrail_class = getattr(module, candidate_name, None)
        if inspect.isclass(guardrail_class) and issubclass(guardrail_class, Guardrail):
            return guardrail_class(model_id=model_id, **kwargs)

        raise ImportError(f"Could not resolve guardrail class for '{guardrail_module_name}' in {module.__name__}")
