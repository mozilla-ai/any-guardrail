import importlib
import inspect
import re
from typing import Any

from any_guardrail.base import Guardrail, GuardrailName


class AnyGuardrail:
    """Factory class for creating guardrail instances."""

    @classmethod
    def get_supported_guardrails(cls) -> list[GuardrailName]:
        """List all supported guardrails."""
        return list(GuardrailName)

    @classmethod
    def get_supported_model(cls, guardrail_name: GuardrailName) -> list[str]:
        """Get the model IDs supported by a specific guardrail."""
        guardrail_class = cls._get_guardrail_class(guardrail_name)
        return guardrail_class.SUPPORTED_MODELS

    @classmethod
    def get_all_supported_models(cls) -> dict[str, list[str]]:
        """Get all model IDs supported by all guardrails."""
        model_ids = {}
        for guardrail_name in cls.get_supported_guardrails():
            model_ids[guardrail_name.value] = cls.get_supported_model(guardrail_name)
        return model_ids

    @classmethod
    def create(cls, guardrail_name: GuardrailName, **kwargs: Any) -> Guardrail:
        """Create a guardrail instance.

        Args:
            guardrail_name: The name of the guardrail to use.
            **kwargs: Additional keyword arguments to pass to the guardrail constructor.

        Returns:
            A guardrail instance.

        """
        guardrail_class = cls._get_guardrail_class(guardrail_name)
        return guardrail_class(**kwargs)

    @classmethod
    def _get_guardrail_class(cls, guardrail_name: GuardrailName) -> type[Guardrail]:
        guardrail_module_name = f"{guardrail_name.value}"
        module_path = f"any_guardrail.guardrails.{guardrail_module_name}.{guardrail_module_name}"

        module = importlib.import_module(module_path)
        parts = re.split(r"[^A-Za-z0-9]+", guardrail_module_name)
        candidate_name = "".join(p.capitalize() for p in parts if p)
        guardrail_class = getattr(module, candidate_name, None)
        if inspect.isclass(guardrail_class) and issubclass(guardrail_class, Guardrail):
            return guardrail_class
        msg = f"Could not resolve guardrail class for '{guardrail_module_name}' in {module.__name__}"
        raise ImportError(msg)
