"""Shared post-processing for the off-topic cross-encoder implementations."""

from typing import Any

import torch

from any_guardrail.types import CategoryResult, GuardrailOutput


def off_topic_output(logits: Any) -> GuardrailOutput[bool, dict[str, float], float]:
    """Map cross-encoder logits to the standard output shape.

    Args:
        logits: Raw model output tensor of shape ``(1, 2)`` where index 0 is
            the on-topic class and index 1 is the off-topic class.

    Returns:
        GuardrailOutput with valid=True when on-topic, ``score`` = P(off-topic)
        (canonical risk direction), and both class probabilities in ``categories``.

    """
    probabilities = torch.softmax(logits, dim=1)
    predicted_label = int(torch.argmax(probabilities, dim=1).item())
    on_topic_probability, off_topic_probability = (float(p) for p in probabilities.cpu().numpy().tolist()[0])
    off_topic = predicted_label == 1  # label '1' indicates off-topic
    return GuardrailOutput(
        valid=not off_topic,
        score=off_topic_probability,
        categories=[
            CategoryResult(name="on-topic", score=on_topic_probability),
            CategoryResult(name="off-topic", score=off_topic_probability, triggered=off_topic),
        ],
    )
