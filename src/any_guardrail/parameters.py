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
    """Whether the parameter must be *supplied as this argument* — i.e. the signature has no
    default. This is a property of the function signature, not of what the model needs at
    runtime: a value that also has an env-var fallback (see :attr:`env_var`) or is one of a
    :class:`RequirementGroup` reads ``required = False`` here even though a value is mandatory.
    For "must a value exist to run at all", consult :attr:`effectively_required` and the
    guardrail's requirement groups instead."""

    default: Any = None
    """The signature default in JSON-native form, or ``None`` when required or unset."""

    effectively_required: bool = False
    """Whether a value for this parameter must exist at runtime for the guardrail to run, even
    though the signature gives it a default (so :attr:`required` is ``False``). ``True`` when the
    guardrail raises if the value resolves from neither the argument nor its :attr:`env_var`. This
    is the field a config UI should gate its required-marker on for single-source settings.
    Members of a one-of :class:`RequirementGroup` are ``False`` here — the group carries their
    (collective) requirement instead."""

    env_var: str | None = None
    """The environment variable that supplies this parameter's value when the argument is not
    passed (e.g. ``"ALINIA_ENDPOINT"``), or ``None`` when there is no env-var fallback. Lets a UI
    tell the user "provide this, or set ``$ENV``"."""

    secret: bool = False
    """Whether this parameter is a credential (API key, token, secret access key, credential-
    bearing client/session). A UI should render it as a masked field and must not log or persist
    it in plain text."""

    choices: tuple[str, ...] | None = None
    """Allowed values for ``enum`` parameters (e.g. ``model_id`` from ``SUPPORTED_MODELS``,
    ``criteria`` from the content registry); ``None`` for non-enum parameters."""

    description: str | None = None
    """One-line description parsed from the guardrail's docstring, when available."""


class RequirementGroup(BaseModel):
    """A guardrail-level "at least one of these must be provided" constraint.

    Some guardrails require *a value* that no single parameter's :attr:`ParameterSpec.required`
    or :attr:`ParameterSpec.effectively_required` can express, because it can be satisfied by any
    of several parameters — e.g. watsonx needs a ``project_id`` *or* a ``space_id``. Each group
    names the interchangeable parameters (and any environment variables that also satisfy it); a
    config UI should require the user to supply at least one member.
    """

    model_config = ConfigDict(frozen=True)

    description: str
    """Human-readable statement of the constraint, e.g. ``"A project or space is required."``"""

    parameters: tuple[str, ...]
    """The interchangeable ``create``/``validate`` parameter names — at least one must be given
    (unless satisfied by one of :attr:`env_vars`)."""

    env_vars: tuple[str, ...] = ()
    """Environment variables that also satisfy the group (e.g. ``("WATSONX_PROJECT_ID",
    "WATSONX_SPACE_ID")``); empty when the group can only be satisfied by passing an argument."""
