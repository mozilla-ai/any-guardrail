"""Static taxonomy and capability metadata for guardrails.

This module is deliberately dependency-free (only the standard library and
Pydantic): it defines the enums and the :class:`GuardrailMetadata` model that
describe *what a guardrail is designed to detect* and *how it runs*, without
importing any guardrail implementation, ``torch``, or ``transformers``. That
keeps discovery/filtering (see :class:`any_guardrail.api.AnyGuardrail`) cheap
and importable in environments that only install a subset of the backends.

This capability metadata is a different, static axis from the per-call results
in :class:`any_guardrail.types.GuardrailOutput`: ``GuardrailOutput.categories``
records what a single ``validate()`` call *found*, whereas the metadata here
records what a guardrail *can* find and the shape of its decision.
"""

from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator


class GuardrailCategory(StrEnum):
    """What a guardrail is designed to detect (a guardrail may span several)."""

    PROMPT_INJECTION = "prompt_injection"
    """Prompt injection, jailbreak, and instruction-override attempts."""

    CONTENT_SAFETY = "content_safety"
    """Harmful content: violence, sexual, self-harm, dangerous, or criminal material."""

    TOXICITY = "toxicity"
    """Hate, harassment, and profanity."""

    PII = "pii"
    """Personal / sensitive-data detection."""

    HALLUCINATION = "hallucination"
    """Groundedness / RAG-faithfulness of a response against provided context."""

    OFF_TOPIC = "off_topic"
    """Topical relevance / answer relevance."""

    BIAS = "bias"
    """Social bias / fairness."""

    TOOL_USE = "tool_use"
    """Function-calling / agent-action validity."""

    GENERAL_JUDGE = "general_judge"
    """Open-ended rubric / quality scoring against bring-your-own criteria."""


class GuardrailStage(StrEnum):
    """Where in a request/response flow a guardrail runs.

    A guardrail that screens both the prompt and the response has ``stages ==
    {INPUT, OUTPUT}`` (there is no separate ``EITHER`` value). ``RAG_CONTEXT``
    marks guardrails that additionally consume retrieved documents/context.
    """

    INPUT = "input"
    """Screens the user prompt (pre-call)."""

    OUTPUT = "output"
    """Screens the model response (post-call)."""

    RAG_CONTEXT = "rag_context"
    """Consumes retrieved documents/context (e.g. groundedness checks)."""


class OutputShape(StrEnum):
    """The decision form a guardrail produces (aligns with the populated ``GuardrailOutput`` fields)."""

    BINARY = "binary"
    """A single flagged / not-flagged verdict."""

    MULTI_LABEL = "multi_label"
    """Independent per-category scores/verdicts."""

    CATEGORICAL = "categorical"
    """A taxonomy verdict (e.g. Llama Guard S-codes)."""

    SCORE = "score"
    """A scalar risk score."""

    RUBRIC = "rubric"
    """A judge score against a rubric (e.g. 1-5 / 1-10)."""

    SPAN = "span"
    """Character-offset spans (e.g. hallucination or PII spans)."""


class BackendType(StrEnum):
    """How a guardrail executes."""

    LOCAL_ENCODER = "local_encoder"
    """A local encoder classifier (HuggingFace or encoderfile)."""

    LOCAL_DECODER = "local_decoder"
    """A local decoder LLM (HuggingFace or llamafile)."""

    HOSTED_API = "hosted_api"
    """A hosted service requiring a key/endpoint."""

    LIBRARY_WRAPPED = "library_wrapped"
    """A third-party Python library invoked directly (its own optional extra)."""


class VariantLicense(BaseModel):
    """License governing a single model variant of a guardrail.

    Used where a guardrail's ``SUPPORTED_MODELS`` span several base models with
    different governing licenses (e.g. Llama Guard's 3.2 / 3.1 / 4 variants, or
    PolyGuard's non-commercial Ministral vs Apache Qwen variants), so a single
    ``default_license`` string cannot capture per-variant redistribution terms.
    Instances are frozen, so a ``tuple`` of them keeps :class:`GuardrailMetadata`
    hashable.
    """

    model_config = ConfigDict(frozen=True)

    model_id: str
    """The variant's model ID (one of the guardrail's ``SUPPORTED_MODELS``)."""

    license: str
    """SPDX-ish license governing this variant (e.g. ``"llama-3.2"``, ``"mrl"``, ``"apache-2.0"``)."""


class GuardrailMetadata(BaseModel):
    """Static, queryable capability metadata for a single guardrail.

    Instances are frozen (hashable, immutable) and live in the import-free
    registry ``any_guardrail.registry.GUARDRAIL_METADATA``; each guardrail class
    also exposes the same instance as ``METADATA``.
    """

    model_config = ConfigDict(frozen=True)

    description: str
    """One-sentence, user-facing summary. Equals the guardrail class docstring's first line."""

    display_name: str
    """Human-facing title used in docs/navigation (e.g. ``"GLiGuard"``, ``"Prompt Guard 2"``)."""

    categories: frozenset[GuardrailCategory]
    """Everything this guardrail is designed to detect (see :class:`GuardrailCategory`)."""

    primary_category: GuardrailCategory
    """The guardrail's headline category (must be one of ``categories``).

    Used to place each guardrail under exactly one section in grouped docs/navigation,
    where a multi-category guardrail would otherwise appear several times.
    """

    stages: frozenset[GuardrailStage]
    """Where it runs; ``{INPUT, OUTPUT}`` means it screens both prompt and response."""

    output_shapes: frozenset[OutputShape]
    """The decision form(s) it can produce."""

    backend: BackendType
    """How it executes."""

    required_validate_kwargs: frozenset[str] = Field(default_factory=frozenset)
    """``validate()`` arguments that must be supplied (beyond the primary text)."""

    optional_validate_kwargs: frozenset[str] = Field(default_factory=frozenset)
    """``validate()`` arguments that may be supplied (e.g. ``output_text``, ``documents``)."""

    requires_api_key: bool = False
    """Whether the guardrail needs an API key / credential to run."""

    multilingual: bool = False
    """Whether the guardrail is designed for more than English."""

    multimodal: bool = False
    """Whether the guardrail accepts non-text input (e.g. images)."""

    vendor: str
    """Organization that produced the model/service (e.g. ``"IBM"``, ``"Meta"``)."""

    default_license: str
    """SPDX-ish license of the default model/service (i.e. of ``SUPPORTED_MODELS[0]``;
    e.g. ``"apache-2.0"``, ``"proprietary"``). See ``variant_licenses`` for the per-variant breakdown."""

    variant_licenses: tuple[VariantLicense, ...] = ()
    """Per-variant licenses, for guardrails whose ``SUPPORTED_MODELS`` span base models with
    different governing licenses (empty when ``default_license`` covers every variant). Each entry's
    ``model_id`` is one of the guardrail's ``SUPPORTED_MODELS``. This is the redistribution-governing
    metadata a downstream consumer reads to decide per-variant eligibility."""

    @model_validator(mode="after")
    def _primary_in_categories(self) -> Self:
        """Ensure ``primary_category`` is one of ``categories``."""
        if self.primary_category not in self.categories:
            msg = f"primary_category {self.primary_category!r} must be in categories {sorted(self.categories)}"
            raise ValueError(msg)
        return self

    @field_serializer(
        "categories",
        "stages",
        "output_shapes",
        "required_validate_kwargs",
        "optional_validate_kwargs",
    )
    def _serialize_set(self, value: frozenset[Any]) -> list[str]:
        """Emit set-valued fields as sorted string lists so JSON export is deterministic.

        Applies to both the ``str`` kwarg sets and the ``StrEnum`` sets (categories,
        stages, output_shapes); each element is stringified to its value explicitly.
        """
        return sorted(str(member) for member in value)

    @field_serializer("variant_licenses")
    def _serialize_variant_licenses(self, value: tuple[VariantLicense, ...]) -> list[dict[str, str]]:
        """Emit per-variant licenses as a ``model_id``-sorted list so JSON export is deterministic."""
        return [
            {"model_id": variant.model_id, "license": variant.license}
            for variant in sorted(value, key=lambda v: v.model_id)
        ]
