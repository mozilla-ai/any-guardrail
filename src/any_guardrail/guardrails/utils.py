from typing import Any

import numpy as np

from any_guardrail.types import GuardrailInferenceOutput, GuardrailOutput

# Type alias for standard HuggingFace dict types
HFDict = dict[str, Any]


def _softmax(_outputs):  # type: ignore[no-untyped-def]
    maxes = np.max(_outputs, axis=-1, keepdims=True)
    shifted_exp = np.exp(_outputs - maxes)
    return shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)


def match_injection_label(
    model_outputs: GuardrailInferenceOutput[HFDict], injection_label: str, id2label: dict[int, str]
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
