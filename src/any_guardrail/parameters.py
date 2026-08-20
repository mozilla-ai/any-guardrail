"""Machine-readable parameter schema for guardrails (issue #206).

Where :mod:`any_guardrail.taxonomy` describes *what a guardrail detects*, this module
describes *what parameters it takes* — a typed, import-free, JSON-exportable schema for both
the ``create`` (``__init__``) and ``validate`` parameters of every guardrail, so a downstream
consumer can auto-generate config UIs without importing the package's model backends.

The per-guardrail specs are generated from the guardrails' signatures + docstrings by
``scripts/generate_parameter_data.py`` into the stdlib-only leaf
``any_guardrail._parameter_data``, and assembled into typed :class:`ParameterSpec`s by the
import-free registry :mod:`any_guardrail.parameter_registry`.

One thing a signature cannot express is what a nested value *looks like*: an annotation of
``list[dict]`` or ``str | dict`` classifies as :attr:`ParameterType.JSON`, which tells a config UI
only that a JSON editor is needed. :class:`ParameterShape` and its companions
(:class:`ParameterField`, :class:`ParameterOption`, :class:`ParameterPreset`) carry that missing
structure so those parameters can be rendered as real controls. They are hand-authored per
parameter in :mod:`any_guardrail._authored_parameter_shape_data` and layered onto the generated
specs by the registry. All of it is optional and additive: a consumer that reads only
:attr:`ParameterSpec.type` behaves exactly as it did before shapes existed.

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


class ParameterShape(StrEnum):
    """The concrete shape of a :attr:`ParameterType.JSON` value.

    :attr:`ParameterType.JSON` says only "this is not a flat scalar", which leaves a config UI no
    option but a free-text JSON editor. A shape says what the nesting actually looks like, so the
    same parameter can be rendered as a real control. Declared per parameter in
    :mod:`any_guardrail._authored_parameter_shape_data`; ``None`` on a spec means "no shape is
    declared", which a consumer should treat exactly as it treated ``JSON`` before shapes existed.

    Shapes describe *structure*, never a closed vocabulary. The option and field vocabularies that
    accompany them (see :class:`ParameterOption`, :class:`ParameterField`) are harvested from vendor
    documentation and are suggestions, not validation: a provider may accept values this package has
    never heard of, so a UI must keep a way to supply one.
    """

    STRING_LIST = "string_list"
    """A ``list[str]`` — e.g. blocklist names, context documents."""

    STRING_MAP = "string_map"
    """A flat ``dict[str, str]`` — e.g. observability tags or request metadata."""

    OBJECT_LIST = "object_list"
    """A ``list[dict]`` of uniform records; the record's columns are in
    :attr:`ParameterSpec.item_fields`."""

    OPTION_MAP = "option_map"
    """A ``dict`` keyed by a known set of switches, each either a bool or a small settings dict; the
    keys are in :attr:`ParameterSpec.options`."""

    OPAQUE = "opaque"
    """A live Python object (a configured SDK client or session). It cannot be expressed as
    configuration at all, so a UI must not offer a field for it."""


class ParameterField(BaseModel):
    """One named sub-value of a structured parameter.

    Used for the columns of an :attr:`ParameterShape.OBJECT_LIST` record
    (:attr:`ParameterSpec.item_fields`) and for a parameter's scalar alternative
    (:attr:`ParameterSpec.scalar_alternative`).
    """

    model_config = ConfigDict(frozen=True)

    key: str
    """The key this value takes in the emitted dict (or the whole value, for a scalar alternative)."""

    label: str
    """Human-readable label for the field."""

    required: bool = False
    """Whether a record is meaningless without this key."""

    suggestions: tuple[str, ...] = ()
    """Known-good values, as an *open* hint: a UI should offer these while still accepting
    anything the user types. Empty when there is no useful vocabulary to suggest."""

    choices: tuple[str, ...] | None = None
    """A genuinely closed set of allowed values, when one exists; ``None`` otherwise. Distinct from
    :attr:`suggestions`, which never constrains input."""

    description: str | None = None
    """One-line description of the field."""


class ParameterOption(BaseModel):
    """One switchable key of an :attr:`ParameterShape.OPTION_MAP` parameter."""

    model_config = ConfigDict(frozen=True)

    value: str
    """The key this option takes in the emitted dict."""

    label: str
    """Human-readable label for the option."""

    description: str | None = None
    """One-line description of what enabling this option does."""

    knob: str | None = None
    """The name of a single numeric setting this option carries when enabled (e.g. ``"threshold"``),
    emitted as ``{option: {knob: value}}``. ``None`` when the option is a plain on/off, emitted as
    ``{option: True}``."""

    knob_min: float | None = None
    """Inclusive lower bound for :attr:`knob`, when known."""

    knob_max: float | None = None
    """Inclusive upper bound for :attr:`knob`, when known."""


class ParameterPreset(BaseModel):
    """A ready-made value that configures a parameter for one recognisable risk.

    Presets are what let a UI ask "what do you want to catch?" instead of "what JSON do you want to
    send?". For an :attr:`ParameterShape.OBJECT_LIST`, a preset's :attr:`value` is one record of the
    list, so selecting several presets composes into one list.
    """

    model_config = ConfigDict(frozen=True)

    label: str
    """Human-readable name of the risk this preset addresses, e.g. ``"Prompt injection"``."""

    value: Any
    """The value fragment to emit, e.g. ``{"evaluator": "judge", "criteria":
    "patronus:prompt-injection"}``."""

    category: str | None = None
    """The :class:`~any_guardrail.taxonomy.GuardrailCategory` value this preset addresses, so a UI
    can line presets up with the categories it already filters by. Held as a plain string to keep
    this module a stdlib+pydantic leaf; ``None`` when no taxonomy category fits."""

    description: str | None = None
    """One-line description of what the preset checks."""


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

    shape: ParameterShape | None = None
    """For a :attr:`ParameterType.JSON` parameter, the concrete shape of its value, so a config UI
    can render a control instead of a JSON editor. ``None`` for every other type, and for a ``JSON``
    parameter whose shape is not declared. Never a substitute for :attr:`type`: a consumer that
    ignores this field keeps working exactly as before."""

    item_fields: tuple[ParameterField, ...] = ()
    """The columns of one record, for :attr:`ParameterShape.OBJECT_LIST`; empty otherwise."""

    options: tuple[ParameterOption, ...] = ()
    """The switchable keys, for :attr:`ParameterShape.OPTION_MAP`; empty otherwise."""

    presets: tuple[ParameterPreset, ...] = ()
    """Ready-made values for recognisable risks (see :class:`ParameterPreset`); empty when the
    parameter has no useful presets."""

    scalar_alternative: ParameterField | None = None
    """Set when the parameter also accepts a plain string carrying a *different* meaning from the
    structured form — e.g. alinia's ``detection_config`` takes either an inline detection dict or the
    id of a detection configuration registered with the provider. ``None`` when the structured shape
    is the only accepted form."""


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
