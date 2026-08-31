"""Mapping from any-guardrail model IDs to their encoderfile artifacts on HuggingFace.

Each model has its own HF repo, indexed by the
`mozilla-ai/encoderfiles <https://huggingface.co/collections/mozilla-ai/encoderfiles>`_
collection. A repo holds four flat binaries at its root, one per platform tag::

    {basename}.{platform_tag}.encoderfile

The older shared ``mozilla-ai/encoderfile`` repo (one repo, models nested under
``{subdir}/{basename}/``) is deprecated at HuggingFace's request — see
mozilla-ai/any-guardrail#209. Repo IDs are listed explicitly rather than derived from the
model_id, because the casing does not round-trip: ``ProtectAI/...`` publishes under
``mozilla-ai/protectai-...`` while ``JasperLS`` and ``DuoGuard`` keep theirs.

Power users can bypass this map with ``binary_path=`` (a locally built encoderfile), or
redirect just the repo with ``encoderfile_repo=`` (a fork or mirror), on
:class:`~any_guardrail.providers.encoderfile.EncoderfileProvider`.
"""

# Mapping: any-guardrail model_id -> (hf_repo_id, basename)
#
# Encoders deliberately absent from this map, and why:
#
# Non-redistributable (see ``default_license`` in registry.py):
#   qualifire/prompt-injection-sentinel   elastic-2.0, gated; the artifact was withdrawn
#                                         upstream pending license clearance
#   meta-llama/Llama-Prompt-Guard-2-*     Llama 4 Community, gated
#
# Not externally servable — these read ``provider.tokenizer`` / ``provider.device`` directly,
# so no encoderfile of them could work regardless of license:
#   hbseong/HarmAug-Guard                       (HarmGuard)
#   mozilla-ai/*-off-topic                      (OffTopic; both variants)
#
# Permissible and servable, but no artifact built yet:
#   leolee99/InjecGuard                         legacy sibling of PIGuard
#   DuoGuard/DuoGuard-1.5B-transfer             apache-2.0
#   DuoGuard/DuoGuard-1B-Llama-3.2-transfer     would ship under Llama 3.2 Community terms
#   speakleash/Bielik-Guard-0.{1,5}B-v1.1       apache-2.0
ENCODERFILE_ARTIFACTS: dict[str, tuple[str, str]] = {
    "DuoGuard/DuoGuard-0.5B": (
        "mozilla-ai/DuoGuard-DuoGuard-0.5B-encoderfile",
        "DuoGuard-0.5B",
    ),
    "JasperLS/deberta-v3-base-injection": (
        "mozilla-ai/JasperLS-deberta-v3-base-injection-encoderfile",
        "deberta-v3-base-injection",
    ),
    "JasperLS/gelectra-base-injection": (
        "mozilla-ai/JasperLS-gelectra-base-injection-encoderfile",
        "gelectra-base-injection",
    ),
    "ProtectAI/deberta-v3-base-prompt-injection": (
        "mozilla-ai/protectai-deberta-v3-base-prompt-injection-encoderfile",
        "deberta-v3-base-prompt-injection",
    ),
    "ProtectAI/deberta-v3-base-prompt-injection-v2": (
        "mozilla-ai/protectai-deberta-v3-base-prompt-injection-v2-encoderfile",
        "deberta-v3-base-prompt-injection-v2",
    ),
    "ProtectAI/deberta-v3-small-prompt-injection-v2": (
        "mozilla-ai/protectai-deberta-v3-small-prompt-injection-v2-encoderfile",
        "deberta-v3-small-prompt-injection-v2",
    ),
    "ProtectAI/distilroberta-base-rejection-v1": (
        "mozilla-ai/protectai-distilroberta-base-rejection-v1-encoderfile",
        "distilroberta-base-rejection-v1",
    ),
    "dcarpintero/pangolin-guard-base": (
        "mozilla-ai/dcarpintero-pangolin-guard-base-encoderfile",
        "pangolin-guard-base",
    ),
    "deepset/deberta-v3-base-injection": (
        "mozilla-ai/deepset-deberta-v3-base-injection-encoderfile",
        "deberta-v3-base-injection",
    ),
    "leolee99/PIGuard": (
        "mozilla-ai/leolee99-PIGuard-encoderfile",
        "PIGuard",
    ),
}


def resolve_artifact(model_id: str, platform_tag: str) -> tuple[str, str]:
    """Resolve the HuggingFace repo + filename for the encoderfile of ``model_id``.

    Args:
        model_id: any-guardrail model identifier (matches a SUPPORTED_MODELS entry on a guardrail).
        platform_tag: e.g. ``"aarch64-apple-darwin"`` or ``"x86_64-linux-gnu"``.

    Returns:
        ``(repo_id, filename)`` suitable for ``huggingface_hub.hf_hub_download``.

    Raises:
        KeyError: If ``model_id`` has no published encoderfile artifact.

    """
    if model_id not in ENCODERFILE_ARTIFACTS:
        available = ", ".join(sorted(ENCODERFILE_ARTIFACTS))
        msg = (
            f"No encoderfile artifact registered for model_id {model_id!r}. "
            f"Available: {available}. "
            f"Pass `binary_path=` to use a locally-built encoderfile."
        )
        raise KeyError(msg)
    repo_id, basename = ENCODERFILE_ARTIFACTS[model_id]
    return repo_id, f"{basename}.{platform_tag}.encoderfile"
