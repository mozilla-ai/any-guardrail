from any_guardrail.types import AnyDict, GuardrailInferenceOutput, GuardrailOutput


def default(model_id: str | None, supported_models: list[str]) -> str:
    """Resolve and validate model_id against supported models.

    Args:
        model_id: The model ID provided by the user, or None to use the default.
        supported_models: List of supported model IDs.

    Returns:
        The resolved model ID (first supported model if None was provided).

    Raises:
        ValueError: If model_id is not in supported_models.

    """
    resolved_id = model_id or supported_models[0]
    if resolved_id not in supported_models:
        msg = f"Only supports {supported_models}. Please use this path to instantiate model."
        raise ValueError(msg)
    return resolved_id


def match_injection_label(
    model_outputs: GuardrailInferenceOutput[AnyDict], injection_label: str
) -> GuardrailOutput[bool, None, float]:
    """Match injection label from model outputs.

    Args:
        model_outputs: Inference output with the uniform shape produced by providers
            (``predicted_labels`` and ``scores`` keys).
        injection_label: The label indicating injection/unsafe content.

    Returns:
        GuardrailOutput with valid=True if content is safe, valid=False if injection detected.

    """
    label = model_outputs.data["predicted_labels"][0]
    score = float(model_outputs.data["scores"][0].max())
    return GuardrailOutput(valid=label != injection_label, score=score)


def match_injection_label_batch(
    model_outputs: GuardrailInferenceOutput[AnyDict], injection_label: str
) -> list[GuardrailOutput[bool, None, float]]:
    """Match injection label for a batch of model outputs.

    Args:
        model_outputs: Inference output with the uniform shape produced by providers
            (``predicted_labels`` and ``scores`` keys, batched).
        injection_label: The label indicating injection/unsafe content.

    Returns:
        A list of GuardrailOutputs, one per input in the batch, in the same order.

    """
    predicted_labels = model_outputs.data["predicted_labels"]
    scores = model_outputs.data["scores"]
    return [
        GuardrailOutput(valid=label != injection_label, score=float(row.max()))
        for label, row in zip(predicted_labels, scores, strict=True)
    ]
