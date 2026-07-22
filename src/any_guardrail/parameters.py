"""Machine-readable parameter schema for guardrails (issue #206).

Where :mod:`any_guardrail.taxonomy` describes *what a guardrail detects*, this module
describes *what parameters it takes* — a typed, import-free, JSON-exportable schema for both
the ``create`` (``__init__``) and ``validate`` parameters of every guardrail, so a downstream
consumer can auto-generate config UIs without importing the package's model backends.

The per-guardrail specs are generated from the guardrails' signatures + docstrings by
``scripts/generate_parameter_data.py`` into the stdlib-only leaf
``any_guardrail._parameter_data``, and assembled into typed :class:`ParameterSpec`s by the
import-free registry :mod:`any_guardrail.parameter_registry`.

This module is deliberately dependency-free (only the standard library and Pydantic).
"""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict


class ParameterStage(StrEnum):
    """Which call a parameter belongs to."""

    CREATE = "create"
    """A constructor parameter (``AnyGuardrail.create(...)`` / ``__init__``)."""

    VALIDATE = "validate"
    """A ``validate()`` parameter (beyond the primary input text)."""


class ParameterType(StrEnum):
    """The value shape of a parameter, so a form can render the right control."""

    STRING = "string"
    """A free-text string."""

    INTEGER = "integer"
    """A whole number."""

    NUMBER = "number"
    """A real number (float)."""

    BOOLEAN = "boolean"
    """A true/false flag."""

    ENUM = "enum"
    """A closed set of string choices (see :attr:`ParameterSpec.choices`)."""

    JSON = "json"
    """A nested dict / list / list-of-dict — the "not flat-form-able, use a JSON editor" signal."""


class ParameterSpec(BaseModel):
    """A single typed parameter of a guardrail's ``create`` or ``validate`` call.

    Instances live in the import-free registry
    :data:`any_guardrail.parameter_registry.PARAMETER_REGISTRY`; a public accessor
    ``AnyGuardrail.get_parameter_schema(name)`` returns them per guardrail.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    """Parameter name (as it appears in the signature)."""

    stage: ParameterStage
    """Whether it is a ``create`` (constructor) or ``validate`` parameter."""

    type: ParameterType
    """The value shape (see :class:`ParameterType`)."""

    required: bool
    """Whether the parameter must be supplied (has no default in the signature)."""

    default: Any = None
    """The signature default in JSON-native form, or ``None`` when required or unset."""

    choices: tuple[str, ...] | None = None
    """Allowed values for ``enum`` parameters (e.g. ``model_id`` from ``SUPPORTED_MODELS``,
    ``criteria`` from the content registry); ``None`` for non-enum parameters."""

    description: str | None = None
    """One-line description parsed from the guardrail's docstring, when available."""
