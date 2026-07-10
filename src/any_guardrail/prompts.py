"""Prompt-template models for the guardrail prompt registry (issues #20 / #87).

This module is deliberately dependency-free (only the standard library and
Pydantic): it defines the models that describe *what prompt a generative guardrail
runs* — the author-published (or any-guardrail-adapted) template text, its
placeholders, and its provenance — without importing any guardrail implementation,
``torch``, or ``transformers``. That keeps prompt discovery (see
:meth:`any_guardrail.api.AnyGuardrail.get_prompt`) and the JSON export cheap and
importable in a bare install with no model loaded.

The registry that maps each :class:`~any_guardrail.base.GuardrailName` to its
:class:`PromptSpec` lives in :mod:`any_guardrail.prompt_registry` (which imports
``base`` for the enum); this module stays a leaf so the import-free guarantee holds.

Prompts here are a static, catalog axis — *what a guardrail is designed to ask the
model* — complementary to the per-call :class:`any_guardrail.types.GuardrailOutput`
(*what one ``validate()`` call found*) and the capability metadata in
:mod:`any_guardrail.taxonomy`.
"""

from enum import StrEnum
from string import Formatter
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, field_serializer, model_validator


class PromptAssembly(StrEnum):
    """How a prompt template is turned into what the model actually sees."""

    CHAT = "chat"
    """Segments map to chat roles (e.g. ``system`` / ``user``) rendered into a messages list."""

    RAW = "raw"
    """A single string ``.format()``-ed then fed to the model verbatim (no chat template)."""

    ASSEMBLED = "assembled"
    """Constituent fragments the guardrail assembles imperatively (f-strings, chat-template
    injection, or an upstream library). Reference-only: discoverable and pinnable, but not a
    single ``.format()``-overridable template."""


def _parse_placeholders(segments: dict[str, str]) -> frozenset[str]:
    """Return the set of ``{placeholder}`` field roots across all segment strings.

    Uses :class:`string.Formatter`, so escaped braces (``{{`` / ``}}``) are ignored and an
    unbalanced brace raises ``ValueError`` at construction time. Field roots strip any
    ``.attr`` / ``[key]`` access so ``{a.b}`` records ``a``.
    """
    fields: set[str] = set()
    for text in segments.values():
        for _literal, field_name, _spec, _conv in Formatter().parse(text):
            if field_name:
                fields.add(field_name.split(".")[0].split("[")[0])
    return frozenset(fields)


class PromptTemplate(BaseModel):
    """A single, immutable prompt template with its placeholders and provenance.

    Instances live in the import-free registry
    :data:`any_guardrail.prompt_registry.PROMPT_REGISTRY` (as versions of a
    :class:`PromptSpec`) and are also exposed on prompt-bearing guardrail classes as
    ``PROMPT``.
    """

    model_config = ConfigDict(frozen=True)

    segments: dict[str, str]
    """Named template fragments. For ``RAW`` a single entry (e.g. ``{"system": ...}`` or
    ``{"prompt": ...}``); for ``CHAT`` the chat roles (``{"system": ..., "user": ...}``); for
    ``ASSEMBLED`` the constituent fragments the guardrail assembles."""

    variables: frozenset[str] = frozenset()
    """Placeholder names the template expects (derived from ``segments``; any supplied value is
    ignored and recomputed). Exported as a sorted list for deterministic JSON."""

    assembly: PromptAssembly = PromptAssembly.CHAT
    """How the segments become model input (see :class:`PromptAssembly`)."""

    overridable: bool = True
    """Whether callers can swap this prompt at construction/call time. ``False`` for
    reference-only (``ASSEMBLED``) entries whose runtime wording is fixed by the model's chat
    template or an upstream library."""

    source: str | None = None
    """Provenance URL for the template (author model card, repo file, or paper)."""

    provenance: Literal["author", "adapted"] = "author"
    """Whether the text is the model author's published prompt (``"author"``) or authored/adapted
    by any-guardrail (``"adapted"``). Part of the #194 benchmark cohort signal."""

    description: str | None = None
    """One-line, human-facing note (e.g. ``"absolute grading, no reference answer"``)."""

    @model_validator(mode="before")
    @classmethod
    def _derive_variables(cls, data: Any) -> Any:
        """Recompute ``variables`` from ``segments`` so a template's placeholders can't drift."""
        if isinstance(data, dict) and "segments" in data:
            segments = data["segments"]
            if isinstance(segments, dict):
                return {**data, "variables": _parse_placeholders(segments)}
        return data

    @field_serializer("variables")
    def _serialize_variables(self, value: frozenset[str]) -> list[str]:
        """Emit ``variables`` as a sorted list so JSON export is deterministic."""
        return sorted(value)

    def render(self, **values: str) -> dict[str, str]:
        """Return each segment ``.format()``-ed with ``values`` (the flat case).

        Guardrails with nested fragments (e.g. GLIDER) or ``ASSEMBLED`` prompts read
        ``.segments[...]`` directly and assemble themselves instead.
        """
        return {name: text.format(**values) for name, text in self.segments.items()}


class PromptSpec(BaseModel):
    """The set of named prompt versions for one guardrail, with a default.

    A guardrail keeps its *current runtime* prompt as ``default_version`` (so behavior and
    benchmark reproducibility are unchanged) and may carry additional author-published variants
    (e.g. absolute vs. pairwise judging) as further versions.
    """

    model_config = ConfigDict(frozen=True)

    versions: dict[str, PromptTemplate]
    """Named prompt versions. ``(guardrail, version)`` is the pinnable prompt identifier."""

    default_version: str = "default"
    """Key in ``versions`` used when no version is requested (the current runtime prompt)."""

    @model_validator(mode="after")
    def _default_in_versions(self) -> Self:
        """Ensure ``default_version`` names a real version."""
        if self.default_version not in self.versions:
            msg = f"default_version {self.default_version!r} not in versions {sorted(self.versions)}"
            raise ValueError(msg)
        return self

    def resolve(self, version: str | None = None) -> PromptTemplate:
        """Return the requested version, or the default when ``version`` is ``None``."""
        return self.versions[version or self.default_version]
