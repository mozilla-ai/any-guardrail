import importlib
import pkgutil
import inspect
from typing import Any, Dict, Type, List
from .utils.constants import GUARDRAIL_PKG

# Directory containing guardrail implementations
# Cache for guardrail classes
_GUARDRAIL_CLASSES: Dict[str, Type] = {}

def _discover_guardrail_classes() -> None:
    """Dynamically discover all guardrail classes in the guardrails package."""
    global _GUARDRAIL_CLASSES
    if _GUARDRAIL_CLASSES:
        return  # Already discovered
    package = importlib.import_module(GUARDRAIL_PKG)
    for _, modname, ispkg in pkgutil.iter_modules(package.__path__):
        if ispkg:
            continue
        module = importlib.import_module(f"{GUARDRAIL_PKG}.{modname}")
        for name, obj in inspect.getmembers(module, inspect.isclass):
            # Only include classes defined in this module (not imports)
            if obj.__module__ == module.__name__:
                # Store by lowercase name for case-insensitive lookup
                _GUARDRAIL_CLASSES[name.lower()] = obj

def list_guardrail_classes() -> List[str]:
    """Return a list of available guardrail class names."""
    _discover_guardrail_classes()
    return sorted(_GUARDRAIL_CLASSES.keys())

def instantiate_guardrail(name: str, **kwargs) -> Any:
    """
    Instantiate a guardrail class by name, passing kwargs to its constructor.
    Args:
        name: Name of the guardrail class (case-insensitive).
        **kwargs: Arguments to pass to the class constructor.
    Returns:
        An instance of the requested guardrail class.
    Raises:
        ValueError: If the class is not found or instantiation fails.
    """
    _discover_guardrail_classes()
    cls = _GUARDRAIL_CLASSES.get(name.lower())
    if cls is None:
        raise ValueError(f"Guardrail class '{name}' not found. Available: {list_guardrail_classes()}")
    try:
        return cls(**kwargs)
    except TypeError as e:
        raise ValueError(f"Error instantiating '{name}': {e}") 