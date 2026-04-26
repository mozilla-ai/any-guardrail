from pathlib import Path

import numpy as np

from any_guardrail.types import AnyDict, GuardrailInferenceOutput, GuardrailOutput


def default(
    model_id: str | None,
    supported_models: list[str],
    local_dir: str | Path | None = None,
) -> str:
    """Resolve and validate model_id against supported models.

    Args:
        model_id: The model ID provided by the user, or None to use the default.
        supported_models: List of supported model IDs.
        local_dir: Path to a local model directory produced by
            ``huggingface-cli download --local-dir`` or equivalent. When provided,
            model_id is still validated but weights are loaded from disk;
            no network access is required.

    Returns:
        The resolved model ID or absolute local path.

    Raises:
        ValueError: If model_id is not in supported_models, or if local_dir is
            provided without a model_id.

    """
    if local_dir is not None:
        if model_id is None:
            msg = "model_id is required when local_dir is provided."
            raise ValueError(msg)
        if model_id not in supported_models:
            msg = f"Only supports {supported_models}. Please use this path to instantiate model."
            raise ValueError(msg)
        return str(Path(local_dir).resolve())
    resolved_id = model_id or supported_models[0]
    if resolved_id not in supported_models:
        msg = f"Only supports {supported_models}. Please use this path to instantiate model."
        raise ValueError(msg)
    return resolved_id


def _softmax(_outputs):  # type: ignore[no-untyped-def]
    maxes = np.max(_outputs, axis=-1, keepdims=True)
    shifted_exp = np.exp(_outputs - maxes)
    return shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)


def match_injection_label(
    model_outputs: GuardrailInferenceOutput[AnyDict], injection_label: str, id2label: dict[int, str]
) -> GuardrailOutput[bool, None, float]:
    """Match injection label from model outputs.

    Args:
        model_outputs: The wrapped inference output containing logits.
        injection_label: The label indicating injection/unsafe content.
        id2label: Mapping from label IDs to label names.

    Returns:
        GuardrailOutput with valid=True if content is safe, valid=False if injection detected.

    """
    logits = model_outputs.data["logits"][0].numpy()
    scores = _softmax(logits)  # type: ignore[no-untyped-call]
    label = id2label[scores.argmax().item()]
    return GuardrailOutput(valid=label != injection_label, score=scores.max().item())
