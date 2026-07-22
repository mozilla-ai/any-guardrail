"""Central, import-free registry of guardrail parameter schemas (issue #206).

Assembles the generated stdlib-only ``any_guardrail._parameter_data`` into typed, frozen
:class:`~any_guardrail.parameters.ParameterSpec` tuples per guardrail. Imports only
``any_guardrail.base`` (for ``GuardrailName``), ``any_guardrail.parameters`` (leaf models), and
the generated ``any_guardrail._parameter_data`` leaf — never a guardrail implementation — so
parameter discovery never pulls in ``torch`` / ``transformers`` or spins up a backend, and works
in a bare install. The public accessor is ``AnyGuardrail.get_parameter_schema(name)``.
"""

from any_guardrail._parameter_data import PARAMETER_DATA
from any_guardrail.base import GuardrailName
from any_guardrail.parameters import ParameterSpec

PARAMETER_REGISTRY: dict[GuardrailName, tuple[ParameterSpec, ...]] = {
    GuardrailName(name): tuple(ParameterSpec(**spec) for spec in specs) for name, specs in PARAMETER_DATA.items()
}


def get_parameter_schema(name: GuardrailName) -> list[ParameterSpec]:
    """Return the typed ``create`` + ``validate`` parameter specs for a guardrail.

    Returns an empty list for a guardrail that takes no configurable parameters. Reads the
    import-free registry only — no guardrail implementation or model backend is imported.
    """
    return list(PARAMETER_REGISTRY.get(name, ()))
