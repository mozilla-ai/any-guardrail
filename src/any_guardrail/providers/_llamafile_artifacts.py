"""Mapping from any-guardrail model IDs to their llamafile artifacts on HuggingFace.

Unlike encoderfile artifacts, llamafiles are multi-platform (Cosmopolitan Libc),
so there's no per-arch tag — each model_id maps to a single ``(repo_id, filename)``
pair.

Coverage is per *model*, not per guardrail, and two guardrails have a default model with
no published artifact — ``Qwen3Guard`` (defaults to ``Qwen3Guard-Gen-0.6B``) and
``PolyGuard`` (defaults to the non-commercial ``PolyGuard-Ministral``). On those, a bare
``LlamafileProvider()`` raises ``KeyError``; pass an explicit ``model_id=`` that is in the
map below.

Power users can bypass this map by passing ``binary_path=`` (for a local file) or
``repo_id=`` + ``filename=`` (for an unmapped HF artifact) directly to
:class:`~any_guardrail.providers.llamafile.LlamafileProvider`.
"""

# Staging repo for the guardrail llamafile fleet. Each model is expected to move to its
# own HF repo (mozilla-ai/any-guardrail#210), the way the shared ``mozilla-ai/encoderfile``
# repo was split up at HuggingFace's request (#209) — a per-model repo is easier to find and
# gives each artifact somewhere to carry its own license text. When that lands, replace the
# repo id on the affected rows below.
_GUARDRAIL_LLAMAFILE_REPO = "mozilla-ai/llamafile-guardrails"

# Mapping: any-guardrail model_id -> (hf_repo_id, filename)
#
# Decoders deliberately absent from this map, and why:
#
# Non-redistributable (see ``variant_licenses`` / ``default_license`` in registry.py):
#   PatronusAI/glider                        cc-by-nc-4.0, non-commercial
#   ToxicityPrompts/PolyGuard-Ministral      Mistral Research License, non-commercial
#   prometheus-eval/prometheus-{7b,13b}-v1.0 Llama 2 Community terms
#   meta-llama/Llama-Guard-4-12B             Llama 4 Community, gated (not a SUPPORTED_MODEL)
#
# Not externally servable — these can't route through ``generate_chat`` with default flags,
# so a llamafile of them would be unusable regardless of license:
#   allenai/wildguard          needs apply_chat_template=False (ships its own instruction wrapper)
#   kakaocorp/kanana-safeguard-*  needs skip_special_tokens=False (the verdict IS a special token)
#   google/shieldgemma-*       reads Yes/No vocabulary logits; never calls generate_chat
#   Qwen/Qwen3Guard-Stream-*   drives provider.model.stream_moderate_from_ids as remote code
#
# Permissible and servable, but no artifact built yet:
#   meta-llama/Llama-Guard-3-1B, meta-llama/Llama-Guard-3-8B
LLAMAFILE_ARTIFACTS: dict[str, tuple[str, str]] = {
    "AtlaAI/Selene-1-Mini-Llama-3.1-8B": (
        _GUARDRAIL_LLAMAFILE_REPO,
        "Selene-1-Mini-Llama-3.1-8B-Q4_K_M.llamafile",
    ),
    "Qwen/Qwen3Guard-Gen-4B": (
        _GUARDRAIL_LLAMAFILE_REPO,
        "Qwen3Guard-Gen-4B-Q4_K_M.llamafile",
    ),
    "Qwen/Qwen3Guard-Gen-8B": (
        _GUARDRAIL_LLAMAFILE_REPO,
        "Qwen3Guard-Gen-8B-Q4_K_M.llamafile",
    ),
    "ToxicityPrompts/PolyGuard-Qwen": (
        _GUARDRAIL_LLAMAFILE_REPO,
        "PolyGuard-Qwen-Q4_K_M.llamafile",
    ),
    "ToxicityPrompts/PolyGuard-Qwen-Smol": (
        _GUARDRAIL_LLAMAFILE_REPO,
        "PolyGuard-Qwen-Smol-Q4_K_M.llamafile",
    ),
    "ibm-granite/granite-guardian-4.1-8b": (
        _GUARDRAIL_LLAMAFILE_REPO,
        "granite-guardian-4.1-8b-Q4_K_M.llamafile",
    ),
    "nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3": (
        _GUARDRAIL_LLAMAFILE_REPO,
        "Llama-3.1-Nemotron-Safety-Guard-8B-v3-Q4_K_M.llamafile",
    ),
    "openai/gpt-oss-safeguard-20b": (
        _GUARDRAIL_LLAMAFILE_REPO,
        "gpt-oss-safeguard-20b-UD-Q4_K_XL.llamafile",
    ),
    "opencompass/CompassJudger-2-7B-Instruct": (
        _GUARDRAIL_LLAMAFILE_REPO,
        "CompassJudger-2-7B-Instruct-Q4_K_M.llamafile",
    ),
    "prometheus-eval/prometheus-7b-v2.0": (
        _GUARDRAIL_LLAMAFILE_REPO,
        "prometheus-7b-v2.0-Q4_K_M.llamafile",
    ),
    "tomg-group-umd/DynaGuard-8B": (
        _GUARDRAIL_LLAMAFILE_REPO,
        "DynaGuard-8B-Q4_K_M.llamafile",
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
