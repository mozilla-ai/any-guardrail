# Taxonomy

The vocabulary behind guardrail metadata (see the [AnyGuardrail reference](any_guardrail.md) for the `list_guardrails` / `group_by` query API and [Guardrails](guardrails/index.md) for the catalog grouped by primary category).

A machine-readable export of every guardrail's metadata is published at <https://raw.githubusercontent.com/mozilla-ai/any-guardrail/main/schemas/guardrail_metadata.json>.

## GuardrailCategory

What a guardrail is designed to detect (a guardrail may span several).

| Value | Meaning |
|-------|---------|
| `prompt_injection` | Prompt injection, jailbreak, and instruction-override attempts. |
| `content_safety` | Harmful content: violence, sexual, self-harm, dangerous, or criminal material. |
| `toxicity` | Hate, harassment, and profanity. |
| `pii` | Personal / sensitive-data detection. |
| `hallucination` | Groundedness / RAG-faithfulness of a response against provided context. |
| `off_topic` | Topical relevance / answer relevance. |
| `bias` | Social bias / fairness. |
| `tool_use` | Function-calling / agent-action validity. |
| `general_judge` | Open-ended rubric / quality scoring against bring-your-own criteria. |

## GuardrailStage

Where in a request/response flow a guardrail runs.

A guardrail that screens both the prompt and the response has ``stages ==
{INPUT, OUTPUT}`` (there is no separate ``EITHER`` value). ``RAG_CONTEXT``
marks guardrails that additionally consume retrieved documents/context.

| Value | Meaning |
|-------|---------|
| `input` | Screens the user prompt (pre-call). |
| `output` | Screens the model response (post-call). |
| `rag_context` | Consumes retrieved documents/context (e.g. groundedness checks). |

## OutputShape

The decision form a guardrail produces (aligns with the populated ``GuardrailOutput`` fields).

``SCORE`` and ``RUBRIC`` are also the queryable signal for whether
``GuardrailOutput.score`` can ever be populated: a guardrail declaring
**neither** always leaves ``score`` as ``None`` (it only emits a
categorical/binary verdict, not a calibrated risk value). A guardrail
declaring **either** populates ``score`` in the common, successfully-parsed
case, but individual guardrails may still leave it ``None`` in specific
edge cases (e.g. a fail-closed parse-failure path, or a guardrail that
flags something but has nothing to score) — consult the guardrail's own
docstring for those exceptions.

| Value | Meaning |
|-------|---------|
| `binary` | A single flagged / not-flagged verdict. |
| `multi_label` | Independent per-category scores/verdicts. |
| `categorical` | A taxonomy verdict (e.g. Llama Guard S-codes). |
| `score` | A scalar risk score. ``GuardrailOutput.score`` is populated in the common case. |
| `rubric` | A judge score against a rubric (e.g. 1-5 / 1-10). ``GuardrailOutput.score`` is populated in the common case (via the rubric normalized onto the canonical risk axis). |
| `span` | Character-offset spans (e.g. hallucination or PII spans). |

## DeploymentType

Who controls the lifecycle of the process or service that actually runs the model.

Orthogonal to :class:`InterfaceType`: this says *who manages the model artifact and its
startup/shutdown*, not *how a* ``validate()`` *call reaches it*. Replaces the old
``BackendType``, which conflated the two (a ``local_encoder``/``local_decoder`` guardrail
could be a same-process function call via ``HuggingFaceProvider`` *or* a subprocess reached
over HTTP via ``EncoderfileProvider``/``LlamafileProvider`` — same declared backend, very
different deployment).

| Value | Meaning |
|-------|---------|
| `owned` | any-guardrail controls the lifecycle: it downloads the weights/binary, loads or spawns it, and tears it down. Covers both a HuggingFace checkpoint loaded into the current process and a subprocess (``EncoderfileProvider``/``LlamafileProvider``) any-guardrail starts and stops. |
| `external` | A service that exists and runs independently of any-guardrail: a vendor's hosted API (Azure, Bedrock, OpenAI, ...), or a server the caller points a provider at via its own ``base_url=`` (e.g. an encoderfile/llamafile instance the caller runs and manages). |

## InterfaceType

How a ``validate()`` call reaches the thing that runs the model.

Orthogonal to :class:`DeploymentType`: this says *the calling convention*, not *who owns
the other end*. ``IN_MEMORY`` can only pair with ``DeploymentType.OWNED`` — a same-process
function call cannot reach a process any-guardrail doesn't control; ``GuardrailMetadata``
enforces this.

| Value | Meaning |
|-------|---------|
| `in_memory` | A direct function/library call inside the current process — no socket. E.g. a ``transformers``/``torch`` forward pass, or a wrapped library like ``flow_judge``/``gliner2``. |
| `http` | A REST/JSON call over HTTP — to a subprocess any-guardrail spawned on ``localhost`` (``EncoderfileProvider``/``LlamafileProvider``'s default mode) or to a remote host (every ``EXTERNAL`` hosted-API vendor, and those same providers' ``base_url=`` external-server mode). |
| `grpc` | A gRPC call. Not used by any shipped provider today — ``encoderfile`` binaries expose a gRPC endpoint, but ``EncoderfileProvider`` always passes ``--disable-grpc`` — kept for a future provider that speaks it. |

## ModelArchitecture

The neural-network shape of the model a guardrail runs, independent of where/how it runs.

Orthogonal to :class:`DeploymentType`/:class:`InterfaceType`: an encoder classifier and a
decoder LLM can each be owned or external, in-memory or over the wire. This is the part of
the old ``BackendType`` that was never really about deployment mechanics — kept as its own
axis instead of being lost when ``backend`` split into ``deployment_type``/``interface``.

| Value | Meaning |
|-------|---------|
| `encoder` | A transformer encoder with a classification/regression head (BERT/DeBERTa-shaped). |
| `decoder` | A decoder-only LLM invoked via chat/generation. |
| `unknown` | The guardrail delegates to a third-party service whose internal model architecture any-guardrail does not know or depend on (true of nearly every ``EXTERNAL`` guardrail). |

## HardwareRequirement

Compute a guardrail needs to run its own model.

Only meaningful for :attr:`DeploymentType.OWNED` — an ``EXTERNAL`` service's hardware is the
vendor's concern, not the caller's. ``GuardrailMetadata`` enforces this: ``None`` for
``EXTERNAL``, required (non-``None``) for ``OWNED``.

| Value | Meaning |
|-------|---------|
| `cpu` | Runs comfortably on CPU; no accelerator needed. |
| `gpu_optional` | Runs on CPU but is meaningfully faster, or only practical at scale, with a GPU. |
| `gpu_required` | Impractical without a GPU (a multi-billion-parameter decoder LLM). |

## NetworkEgress

Whether a ``validate()`` call leaves the current process, and if so, how far.

Derived from ``deployment_type``/``interface`` (see
:attr:`GuardrailMetadata.network_egress`) rather than authored on each registry entry —
there is no independent fact it could carry that isn't already implied by those two.

| Value | Meaning |
|-------|---------|
| `none` | Same-process function call — :attr:`InterfaceType.IN_MEMORY`. |
| `local` | A subprocess any-guardrail spawned itself; reachable only on ``localhost``. |
| `remote` | A vendor's or caller-managed server; the call leaves the local host. |
