"""Mapping from any-guardrail model IDs to their llamafile artifacts on HuggingFace.

Unlike encoderfile artifacts, llamafiles are multi-platform (Cosmopolitan Libc),
so there's no per-arch tag — each model_id maps to a single ``(repo_id, filename)``
pair.

Power users can bypass this map by passing ``binary_path=`` (for a local file) or
``repo_id=`` + ``filename=`` (for an unmapped HF artifact) directly to
:class:`~any_guardrail.providers.llamafile.LlamafileProvider`.
"""

# Mapping: any-guardrail model_id -> (hf_repo_id, filename)
LLAMAFILE_ARTIFACTS: dict[str, tuple[str, str]] = {
    "ibm-granite/granite-guardian-4.1-8b": (
        "mozilla-ai/llamafile_0.10_alpha",
        "granite-guardian-4.1-8b.Q6_K.llamafile",
    ),
}


def resolve_artifact(model_id: str) -> tuple[str, str]:
    """Resolve the HuggingFace repo + filename for the llamafile of ``model_id``.

    Args:
        model_id: any-guardrail model identifier (matches a SUPPORTED_MODELS entry
            on a guardrail).

    Returns:
        ``(repo_id, filename)`` suitable for ``huggingface_hub.hf_hub_download``.

    Raises:
        KeyError: If ``model_id`` has no published llamafile artifact in the map.

    """
    if model_id not in LLAMAFILE_ARTIFACTS:
        available = ", ".join(sorted(LLAMAFILE_ARTIFACTS))
        msg = (
            f"No llamafile artifact registered for model_id {model_id!r}. "
            f"Available: {available}. "
            f"Pass `binary_path=`, or `repo_id=` + `filename=`, to "
            f"LlamafileProvider to use an unmapped artifact."
        )
        raise KeyError(msg)
    return LLAMAFILE_ARTIFACTS[model_id]
