"""Central, import-free registry of guardrail prompt templates (issues #20 / #87).

This module is the single source of truth for the default/author-published prompts of
prompt-bearing guardrails. It imports only :mod:`any_guardrail.base` (for
``GuardrailName``) and :mod:`any_guardrail.prompts` (leaf models) — never a guardrail
implementation — so prompt discovery never pulls in ``torch``/``transformers`` or spins
up a backend. Each prompt-bearing guardrail class mirrors its entry here as ``PROMPT``.

The prompt *text* lives here (moved out of the guardrail modules), and each guardrail
imports its default from here — a one-way dependency (guardrail → registry) that keeps
the import-free guarantee. ``(guardrail, version)`` is the pinnable prompt identifier
that benchmark tooling (issue #194) records alongside each score.
"""

from any_guardrail.base import GuardrailName
from any_guardrail.prompts import PromptAssembly, PromptSpec, PromptTemplate

# ---------------------------------------------------------------------------
# AnyLlm — the any-guardrail policy-judge protocol prompt. There is no external
# model author (AnyLlm runs any LLM), so this is an ``adapted`` (our own) prompt.
# Moved here verbatim from ``guardrails/any_llm/any_llm.py`` (which now aliases it),
# so behavior is byte-identical.
# ---------------------------------------------------------------------------
_ANYLLM_SYSTEM_PROMPT = """
You are a guardrail designed to ensure that the input text adheres to a specific policy.
Your only task is to validate the input_text, don't try to answer the user query.

Here is the policy: {policy}

You must return the following:

- valid: bool
    If the input text provided by the user doesn't adhere to the policy, you must reject it (mark it as valid=False).

- explanation: str
    A clear explanation of why the input text was rejected or not.

- risk_score: float (0-1)
    How likely the input text is to violate the policy: 0.0 means clearly compliant,
    1.0 means clearly violating.
"""


PROMPT_REGISTRY: dict[GuardrailName, PromptSpec] = {
    GuardrailName.ANYLLM: PromptSpec(
        versions={
            "default": PromptTemplate(
                segments={"system": _ANYLLM_SYSTEM_PROMPT},
                assembly=PromptAssembly.RAW,
                provenance="adapted",
                description=(
                    "any-guardrail policy-judge protocol: grades the input against a natural-language "
                    "policy and returns a valid / explanation / risk_score verdict."
                ),
            ),
        },
    ),
}
"""Prompt catalog keyed by guardrail. One :class:`PromptSpec` per prompt-bearing guardrail."""

PROMPT_BEARING: frozenset[GuardrailName] = frozenset(PROMPT_REGISTRY)
"""The set of guardrails that have a registered prompt (mirrors ``PROMPT_REGISTRY`` keys)."""


def get_prompt(name: GuardrailName, version: str | None = None) -> PromptTemplate:
    """Return a guardrail's prompt template (its default, or a named ``version``).

    Args:
        name: The guardrail to look up.
        version: A specific version key; ``None`` selects the guardrail's ``default_version``.

    Raises:
        KeyError: If ``name`` has no registered prompt (not a prompt-bearing guardrail), or
            ``version`` is not one of its versions.

    """
    spec = PROMPT_REGISTRY.get(name)
    if spec is None:
        msg = (
            f"{name.value!r} has no registered prompt. Prompt-bearing guardrails: "
            f"{sorted(n.value for n in PROMPT_BEARING)}."
        )
        raise KeyError(msg)
    if version is not None and version not in spec.versions:
        msg = f"{name.value!r} has no prompt version {version!r}; available: {sorted(spec.versions)}."
        raise KeyError(msg)
    return spec.resolve(version)


def list_prompt_versions(name: GuardrailName) -> list[str]:
    """Return the sorted version keys registered for ``name`` (empty if it has no prompt)."""
    spec = PROMPT_REGISTRY.get(name)
    return sorted(spec.versions) if spec is not None else []


def resolve_prompt(
    name: GuardrailName,
    prompt: PromptTemplate | None = None,
    version: str | None = None,
) -> PromptTemplate:
    """Resolve the effective prompt for a guardrail call.

    Precedence: an explicit ``prompt`` (inline override) wins; otherwise the registered
    ``version`` (or the guardrail's default) is returned.
    """
    if prompt is not None:
        return prompt
    return get_prompt(name, version)
