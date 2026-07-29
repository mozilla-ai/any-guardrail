"""Central, import-free registry of guardrail parameter schemas (issue #206).

Assembles the generated stdlib-only ``any_guardrail._parameter_data`` into typed, frozen
:class:`~any_guardrail.parameters.ParameterSpec` tuples per guardrail. Imports only
``any_guardrail.base`` (for ``GuardrailName``), ``any_guardrail.parameters`` (leaf models), and
the generated ``any_guardrail._parameter_data`` leaf — never a guardrail implementation — so
parameter discovery never pulls in ``torch`` / ``transformers`` or spins up a backend, and works
in a bare install. The public accessor is ``AnyGuardrail.get_parameter_schema(name)``.
"""

from any_guardrail._parameter_data import PARAMETER_DATA, REQUIREMENT_GROUPS
from any_guardrail.base import GuardrailName
from any_guardrail.parameters import ParameterSpec, RequirementGroup

PARAMETER_REGISTRY: dict[GuardrailName, tuple[ParameterSpec, ...]] = {
    GuardrailName(name): tuple(ParameterSpec(**spec) for spec in specs) for name, specs in PARAMETER_DATA.items()
}

REQUIREMENT_GROUP_REGISTRY: dict[GuardrailName, tuple[RequirementGroup, ...]] = {
    GuardrailName(name): tuple(RequirementGroup(**group) for group in groups)
    for name, groups in REQUIREMENT_GROUPS.items()
}


def get_parameter_schema(name: GuardrailName) -> list[ParameterSpec]:
    """Return the typed ``create`` + ``validate`` parameter specs for a guardrail.

    Returns an empty list for a guardrail that takes no configurable parameters (its registry
    entry is an empty tuple). The registry covers every ``GuardrailName``, so this indexes
    directly — a missing entry is an internal invariant violation and raises ``KeyError`` rather
    than being masked as "no parameters". Reads the import-free registry only; no guardrail
    implementation or model backend is imported.
    """
    return list(PARAMETER_REGISTRY[name])


def get_requirement_groups(name: GuardrailName) -> list[RequirementGroup]:
    """Return the guardrail's one-of :class:`RequirementGroup` constraints (empty when it has none).

    A group means "at least one of these interchangeable parameters (or their env-var fallbacks)
    must be provided" — e.g. watsonx's ``project_id`` / ``space_id``. Single-source runtime
    requirements are not groups; they are carried by :attr:`ParameterSpec.effectively_required`.
    Reads the import-free registry only; no guardrail implementation or model backend is imported.
    """
    return list(REQUIREMENT_GROUP_REGISTRY.get(name, ()))
