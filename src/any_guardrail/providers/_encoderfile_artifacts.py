"""Mapping from any-guardrail model IDs to their encoderfile artifacts on HuggingFace.

The encoderfile binaries published by Mozilla live at:
    https://huggingface.co/mozilla-ai/encoderfile/tree/main

Each model has a per-platform binary named ``{basename}.{platform_tag}.encoderfile``
under ``{subdir}/{basename}/`` (so the basename appears twice in the path).
"""

# Mapping: any-guardrail model_id -> (hf_repo_subdir, basename)
ENCODERFILE_ARTIFACTS: dict[str, tuple[str, str]] = {
    "ProtectAI/deberta-v3-base-prompt-injection-v2": (
        "protectai/deberta-v3-base-prompt-injection-v2",
        "deberta-v3-base-prompt-injection-v2",
    ),
    "ProtectAI/deberta-v3-small-prompt-injection-v2": (
        "protectai/deberta-v3-small-prompt-injection-v2",
        "deberta-v3-small-prompt-injection-v2",
    ),
    "ProtectAI/deberta-v3-base-prompt-injection": (
        "protectai/deberta-v3-base-prompt-injection",
        "deberta-v3-base-prompt-injection",
    ),
    "ProtectAI/distilroberta-base-rejection-v1": (
        "protectai/distilroberta-base-rejection-v1",
        "distilroberta-base-rejection-v1",
    ),
    "JasperLS/deberta-v3-base-injection": (
        "JasperLS/deberta-v3-base-injection",
        "deberta-v3-base-injection",
    ),
    "JasperLS/gelectra-base-injection": (
        "JasperLS/gelectra-base-injection",
        "gelectra-base-injection",
    ),
    "deepset/deberta-v3-base-injection": (
        "deepset/deberta-v3-base-injection",
        "deberta-v3-base-injection",
    ),
    "DuoGuard/DuoGuard-0.5B": (
        "DuoGuard/DuoGuard-0.5B",
        "DuoGuard-0.5B",
    ),
    "qualifire/prompt-injection-sentinel": (
        "qualifire/prompt-injection-sentinel",
        "prompt-injection-sentinel",
    ),
}


def resolve_artifact_path(model_id: str, platform_tag: str) -> str:
    """Resolve the in-repo path of the encoderfile for ``model_id`` on ``platform_tag``.

    Args:
        model_id: any-guardrail model identifier (matches a SUPPORTED_MODELS entry on a guardrail).
        platform_tag: e.g. ``"aarch64-apple-darwin"`` or ``"x86_64-linux-gnu"``.

    Returns:
        The relative path inside the ``mozilla-ai/encoderfile`` HF repo.

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
    subdir, basename = ENCODERFILE_ARTIFACTS[model_id]
    return f"{subdir}/{basename}.{platform_tag}.encoderfile"
