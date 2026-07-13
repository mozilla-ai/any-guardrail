"""Authored-content models for the guardrail content registry (issues #20 / #87).

Where :mod:`any_guardrail.prompts` holds the *templates* a guardrail wraps around input, this
module holds the author-published **fill-in content** a user supplies to those templates — the
ready-made **policies**, **rubrics**, and **criteria** each guardrail's authors publish (e.g.
Google's ShieldGemma harm-type policies, Prometheus's grading rubrics, Granite Guardian's risk
criteria). Storing them here means users can fetch a vetted policy/rubric instead of writing one:

    ShieldGemma(policy=AnyGuardrail.get_policy(GuardrailName.SHIELD_GEMMA, "dangerous_content"))

This module is deliberately dependency-free (only the standard library and Pydantic) so content
discovery never imports a guardrail implementation, ``torch``, or ``transformers``. The registry
that maps each :class:`~any_guardrail.base.GuardrailName` to its content lives in
:mod:`any_guardrail.content_registry`.
"""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict


class ContentKind(StrEnum):
    """The kind of author-published fill-in content."""

    POLICY = "policy"
    """A safety principle / disallowed-content policy fed to a policy-conditioned guardrail
    (e.g. ShieldGemma's ``policy``, gpt-oss-safeguard's system policy)."""

    RUBRIC = "rubric"
    """A scoring rubric (score-level descriptions) fed to a judge guardrail
    (e.g. Prometheus / Selene / Flow-Judge ``rubric``)."""

    CRITERIA = "criteria"
    """A judging criterion / risk definition (e.g. Granite Guardian's risk criteria,
    Flow-Judge's metric criteria)."""


class AuthoredContent(BaseModel):
    """A single, immutable piece of author-published content (a policy, rubric, or criterion).

    Instances live in the import-free registry
    :data:`any_guardrail.content_registry.CONTENT_REGISTRY`.
    """

    model_config = ConfigDict(frozen=True)

    kind: ContentKind
    """Whether this is a policy, rubric, or criteria (see :class:`ContentKind`)."""

    key: str
    """Short identifier, unique per guardrail *within a kind* (e.g. ``"dangerous_content"``,
    ``"response_faithfulness_5point"``). This is what ``get_policy(name, key)`` takes."""

    content: str
    """The ready-to-use content string, passed straight to the guardrail (e.g. as ``policy=`` /
    ``rubric=`` / ``criteria=``)."""

    source: str | None = None
    """Provenance URL for the content (author model card, repo file, or paper)."""

    provenance: Literal["author", "adapted"] = "author"
    """Whether the text is the model author's published content (``"author"``) or authored/adapted
    by any-guardrail (``"adapted"``)."""

    description: str | None = None
    """One-line, human-facing note (e.g. ``"No Dangerous Content (prompt-only)"``)."""
