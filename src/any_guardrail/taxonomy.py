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

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_serializer, model_validator


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
    """The decision form a guardrail produces (aligns with the populated ``GuardrailOutput`` fields).

    ``SCORE`` and ``RUBRIC`` are also the queryable signal for whether
    ``GuardrailOutput.score`` can ever be populated: a guardrail declaring
    **neither** always leaves ``score`` as ``None`` (it only emits a
    categorical/binary verdict, not a calibrated risk value). A guardrail
    declaring **either** populates ``score`` in the common, successfully-parsed
    case, but individual guardrails may still leave it ``None`` in specific
    edge cases (e.g. a fail-closed parse-failure path, or a guardrail that
    flags something but has nothing to score) — consult the guardrail's own
    docstring for those exceptions.
    """

    BINARY = "binary"
    """A single flagged / not-flagged verdict."""

    MULTI_LABEL = "multi_label"
    """Independent per-category scores/verdicts."""

    CATEGORICAL = "categorical"
    """A taxonomy verdict (e.g. Llama Guard S-codes)."""

    SCORE = "score"
    """A scalar risk score. ``GuardrailOutput.score`` is populated in the common case."""

    RUBRIC = "rubric"
    """A judge score against a rubric (e.g. 1-5 / 1-10). ``GuardrailOutput.score`` is populated in the
    common case (via the rubric normalized onto the canonical risk axis)."""

    SPAN = "span"
    """Character-offset spans (e.g. hallucination or PII spans)."""


class DeploymentType(StrEnum):
    """Who controls the lifecycle of the process or service that actually runs the model.

    Orthogonal to :class:`InterfaceType`: this says *who manages the model artifact and its
    startup/shutdown*, not *how a* ``validate()`` *call reaches it*. Replaces the old
    ``BackendType``, which conflated the two (a ``local_encoder``/``local_decoder`` guardrail
    could be a same-process function call via ``HuggingFaceProvider`` *or* a subprocess reached
    over HTTP via ``EncoderfileProvider``/``LlamafileProvider`` — same declared backend, very
    different deployment).
    """

    OWNED = "owned"
    """any-guardrail controls the lifecycle: it downloads the weights/binary, loads or spawns
    it, and tears it down. Covers both a HuggingFace checkpoint loaded into the current process
    and a subprocess (``EncoderfileProvider``/``LlamafileProvider``) any-guardrail starts and
    stops."""

    EXTERNAL = "external"
    """A service that exists and runs independently of any-guardrail: a vendor's hosted API
    (Azure, Bedrock, OpenAI, ...), or a server the caller points a provider at via its own
    ``base_url=`` (e.g. an encoderfile/llamafile instance the caller runs and manages)."""


class InterfaceType(StrEnum):
    """How a ``validate()`` call reaches the thing that runs the model.

    Orthogonal to :class:`DeploymentType`: this says *the calling convention*, not *who owns
    the other end*. ``IN_MEMORY`` can only pair with ``DeploymentType.OWNED`` — a same-process
    function call cannot reach a process any-guardrail doesn't control; ``GuardrailMetadata``
    enforces this.
    """

    IN_MEMORY = "in_memory"
    """A direct function/library call inside the current process — no socket. E.g. a
    ``transformers``/``torch`` forward pass, or a wrapped library like ``flow_judge``/``gliner2``."""

    HTTP = "http"
    """A REST/JSON call over HTTP — to a subprocess any-guardrail spawned on ``localhost``
    (``EncoderfileProvider``/``LlamafileProvider``'s default mode) or to a remote host (every
    ``EXTERNAL`` hosted-API vendor, and those same providers' ``base_url=`` external-server
    mode)."""

    GRPC = "grpc"
    """A gRPC call. Not used by any shipped provider today — ``encoderfile`` binaries expose a
    gRPC endpoint, but ``EncoderfileProvider`` always passes ``--disable-grpc`` — kept for a
    future provider that speaks it."""


class ModelArchitecture(StrEnum):
    """The neural-network shape of the model a guardrail runs, independent of where/how it runs.

    Orthogonal to :class:`DeploymentType`/:class:`InterfaceType`: an encoder classifier and a
    decoder LLM can each be owned or external, in-memory or over the wire. This is the part of
    the old ``BackendType`` that was never really about deployment mechanics — kept as its own
    axis instead of being lost when ``backend`` split into ``deployment_type``/``interface``.
    """

    ENCODER = "encoder"
    """A transformer encoder with a classification/regression head (BERT/DeBERTa-shaped)."""

    DECODER = "decoder"
    """A decoder-only LLM invoked via chat/generation."""

    UNKNOWN = "unknown"
    """The guardrail delegates to a third-party service whose internal model architecture
    any-guardrail does not know or depend on (true of nearly every ``EXTERNAL`` guardrail)."""


class HardwareRequirement(StrEnum):
    """Compute a guardrail needs to run its own model.

    Only meaningful for :attr:`DeploymentType.OWNED` — an ``EXTERNAL`` service's hardware is the
    vendor's concern, not the caller's. ``GuardrailMetadata`` enforces this: ``None`` for
    ``EXTERNAL``, required (non-``None``) for ``OWNED``.
    """

    CPU = "cpu"
    """Runs comfortably on CPU; no accelerator needed."""

    GPU_OPTIONAL = "gpu_optional"
    """Runs on CPU but is meaningfully faster, or only practical at scale, with a GPU."""

    GPU_REQUIRED = "gpu_required"
    """Impractical without a GPU (a multi-billion-parameter decoder LLM)."""


class NetworkEgress(StrEnum):
    """Whether a ``validate()`` call leaves the current process, and if so, how far.

    Derived from ``deployment_type``/``interface`` (see
    :attr:`GuardrailMetadata.network_egress`) rather than authored on each registry entry —
    there is no independent fact it could carry that isn't already implied by those two.
    """

    NONE = "none"
    """Same-process function call — :attr:`InterfaceType.IN_MEMORY`."""

    LOCAL = "local"
    """A subprocess any-guardrail spawned itself; reachable only on ``localhost``."""

    REMOTE = "remote"
    """A vendor's or caller-managed server; the call leaves the local host."""


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


class AlternateDeployment(BaseModel):
    """One additional ``(deployment_type, interface)`` pair a guardrail can run under.

    Via a different ``provider=``, beyond its default pair. See
    :attr:`GuardrailMetadata.alternate_deployments`. Instances are frozen, so a ``tuple`` of
    them keeps :class:`GuardrailMetadata` hashable.
    """

    model_config = ConfigDict(frozen=True)

    deployment_type: DeploymentType
    """The alternate's :class:`DeploymentType`."""

    interface: InterfaceType
    """The alternate's :class:`InterfaceType`."""


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

    deployment_type: DeploymentType
    """Who controls the default path's lifecycle — the path taken when no ``provider`` is
    supplied. See :class:`DeploymentType`."""

    interface: InterfaceType
    """How the default path is called. See :class:`InterfaceType`. ``IN_MEMORY`` implies
    ``deployment_type == DeploymentType.OWNED`` (enforced below): there's no way to reach a
    process this library doesn't control without going over a socket."""

    architecture: ModelArchitecture
    """The default path's model family. See :class:`ModelArchitecture`. Independent of
    ``deployment_type``/``interface`` — it says what kind of network runs, not who runs it or
    how a call reaches it."""

    hardware_requirement: HardwareRequirement | None = None
    """Compute needed for the default path's own model. Required when ``deployment_type ==
    DeploymentType.OWNED``; must be ``None`` when ``EXTERNAL`` (enforced below) — the vendor's
    hardware isn't the caller's concern. See :class:`HardwareRequirement`."""

    alternate_deployments: tuple[AlternateDeployment, ...] = ()
    """Other ``(deployment_type, interface)`` pairs the same guardrail class can run under via
    a different ``provider=``, beyond the default ``deployment_type``/``interface`` pair above.

    Empty for guardrails whose alternate providers stay on the same pair (e.g. swapping one
    ``HuggingFaceProvider`` device/dtype for another). Populated for two real cases in this
    codebase: (1) a guardrail whose ``HuggingFaceProvider`` default (``OWNED``/``IN_MEMORY``)
    also ships a curated ``EncoderfileProvider``/``LlamafileProvider`` binary — swapping to it
    is ``OWNED``/``HTTP`` (any-guardrail still downloads and spawns the binary, just talks to
    it over a socket instead of calling it directly), and pointing that same provider class at
    a ``base_url=`` server the caller runs themselves is a second alternate,
    ``EXTERNAL``/``HTTP``; and (2) ``Susfactor``, whose gated local ONNX default
    (``OWNED``/``IN_MEMORY``) is also reachable through 0DIN's hosted API via
    ``provider=ZeroDinProvider()`` (``EXTERNAL``/``HTTP``).

    ``AnyGuardrail.list_guardrails(deployment_type=..., interface=...)`` and
    ``group_by("deployment_type" | "interface")`` filter on the scalar defaults only, so a
    guardrail is never double-counted; read this field via ``AnyGuardrail.metadata(name)`` to
    discover the alternates. Likewise ``requires_api_key`` describes the default path — an
    alternate external deployment may still need a credential even when the default doesn't."""

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

    supports_batch: bool = False
    """Whether ``validate()`` runs list input through one real batched inference call
    (a shared forward pass / batched ``generate_chat`` call) rather than the default
    per-item loop. Most ``ThreeStageGuardrail`` subclasses accept list input via the
    inherited ``validate()``; some override it to reject lists with ``TypeError``
    instead. This flag only distinguishes real batching from the default loop among
    guardrails that accept list input at all — it says nothing about whether list
    input is accepted in the first place."""

    vendor: str
    """Organization that produced the model/service (e.g. ``"IBM"``, ``"Meta"``)."""

    default_license: str
    """SPDX-ish license of the guardrail's default model or service: the license of
    ``SUPPORTED_MODELS[0]`` for model-backed guardrails, or of the service/library itself for
    hosted-API and library-wrapped guardrails (e.g. ``"apache-2.0"``, ``"proprietary"``). See
    ``variant_licenses`` for the per-variant breakdown."""

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

    @model_validator(mode="after")
    def _in_memory_requires_owned(self) -> Self:
        """Ensure ``IN_MEMORY`` never pairs with ``EXTERNAL``.

        Checked on the default pair and every alternate: a same-process function call
        cannot reach a process this library doesn't control.
        """
        pairs: list[tuple[DeploymentType, InterfaceType, str]] = [
            (self.deployment_type, self.interface, "the default deployment_type/interface"),
            *(
                (alt.deployment_type, alt.interface, f"alternate_deployments[{i}]")
                for i, alt in enumerate(self.alternate_deployments)
            ),
        ]
        for deployment_type, interface, where in pairs:
            if interface == InterfaceType.IN_MEMORY and deployment_type == DeploymentType.EXTERNAL:
                msg = f"{where} declares IN_MEMORY with EXTERNAL, which is impossible"
                raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _hardware_requirement_matches_deployment_type(self) -> Self:
        """``hardware_requirement`` is required for ``OWNED`` and forbidden for ``EXTERNAL``.

        An external service's hardware is the vendor's concern, not the caller's.
        """
        if self.deployment_type == DeploymentType.OWNED and self.hardware_requirement is None:
            msg = "hardware_requirement is required when deployment_type is OWNED"
            raise ValueError(msg)
        if self.deployment_type == DeploymentType.EXTERNAL and self.hardware_requirement is not None:
            msg = (
                f"hardware_requirement must be None when deployment_type is EXTERNAL, got {self.hardware_requirement!r}"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _alternate_deployments_not_repeated(self) -> Self:
        """Ensure ``alternate_deployments`` lists genuine alternatives to the default pair."""
        default = AlternateDeployment(deployment_type=self.deployment_type, interface=self.interface)
        if default in self.alternate_deployments:
            msg = (
                f"the default ({self.deployment_type!r}, {self.interface!r}) pair must not "
                "also appear in alternate_deployments"
            )
            raise ValueError(msg)
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def network_egress(self) -> NetworkEgress:
        """Derived from ``deployment_type``/``interface`` — see :class:`NetworkEgress`.

        Not authored on registry entries: exposed as a field (rather than left for every
        caller to re-derive) because it's the property compliance-minded callers actually
        filter on, even though it carries no fact independent of the two it's built from.
        """
        if self.interface == InterfaceType.IN_MEMORY:
            return NetworkEgress.NONE
        if self.deployment_type == DeploymentType.OWNED:
            return NetworkEgress.LOCAL
        return NetworkEgress.REMOTE

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

    @field_serializer("alternate_deployments")
    def _serialize_alternate_deployments(self, value: tuple[AlternateDeployment, ...]) -> list[dict[str, str]]:
        """Emit alternate deployments as a sorted list of dicts so JSON export is deterministic."""
        return [
            {"deployment_type": str(alt.deployment_type), "interface": str(alt.interface)}
            for alt in sorted(value, key=lambda a: (a.deployment_type, a.interface))
        ]
