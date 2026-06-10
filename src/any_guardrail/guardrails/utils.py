from collections.abc import Sequence

from any_guardrail.types import AnyDict, CategoryResult, GuardrailInferenceOutput, GuardrailOutput


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


def match_injection_label(model_outputs: GuardrailInferenceOutput[AnyDict], injection_label: str) -> GuardrailOutput:
    """Build a GuardrailOutput from a single-row classification result.

    Args:
        model_outputs: Inference output with the uniform shape produced by providers
            (``scores``, ``predicted_indices``, ``predicted_labels`` and optionally
            ``labels`` keys).
        injection_label: The label indicating injection/unsafe content.

    Returns:
        GuardrailOutput with valid=True if content is safe, ``score`` =
        P(injection), and the full label distribution in ``categories``.

    """
    data = model_outputs.data
    return _match_injection_row(
        scores_row=data["scores"][0],
        predicted_label=data["predicted_labels"][0],
        predicted_index=int(data["predicted_indices"][0]),
        labels=data.get("labels"),
        injection_label=injection_label,
    )


def match_injection_label_batch(
    model_outputs: GuardrailInferenceOutput[AnyDict], injection_label: str
) -> list[GuardrailOutput]:
    """Build GuardrailOutputs for a batch of classification results.

    Args:
        model_outputs: Inference output with the uniform shape produced by providers
            (``scores``, ``predicted_indices``, ``predicted_labels`` and optionally
            ``labels`` keys, batched).
        injection_label: The label indicating injection/unsafe content.

    Returns:
        A list of GuardrailOutputs, one per input, in the same order, each with
        ``score`` = P(injection) and the full label distribution in ``categories``.

    """
    data = model_outputs.data
    labels = data.get("labels")
    return [
        _match_injection_row(
            scores_row=row,
            predicted_label=predicted_label,
            predicted_index=int(predicted_index),
            labels=labels,
            injection_label=injection_label,
        )
        for row, predicted_label, predicted_index in zip(
            data["scores"], data["predicted_labels"], data["predicted_indices"], strict=True
        )
    ]


def _match_injection_row(
    scores_row: Sequence[float],
    predicted_label: str,
    predicted_index: int,
    labels: Sequence[str] | None,
    injection_label: str,
) -> GuardrailOutput:
    """Map one row of class probabilities to the standard output shape.

    ``score`` is the probability of the injection class (canonical risk
    direction: higher = riskier), NOT the probability of the predicted class.
    When the provider can't surface the full label list (encoderfile), the
    names are reconstructed for the 2-class case: the predicted index keeps
    its label and the complement index is assumed to be the missing one.
    """
    probabilities = [float(probability) for probability in scores_row]
    names = _resolve_label_names(len(probabilities), labels, predicted_index, predicted_label, injection_label)
    injection_index = names.index(injection_label) if injection_label in names else None
    categories = [
        CategoryResult(
            name=name,
            score=probability,
            triggered=(predicted_label == injection_label) if name == injection_label else None,
        )
        for name, probability in zip(names, probabilities, strict=True)
    ]
    return GuardrailOutput(
        valid=predicted_label != injection_label,
        score=probabilities[injection_index] if injection_index is not None else None,
        categories=categories,
    )


def _resolve_label_names(
    num_classes: int,
    labels: Sequence[str] | None,
    predicted_index: int,
    predicted_label: str,
    injection_label: str,
) -> list[str]:
    """Resolve the ordered class-label names for one classification row."""
    if labels is not None:
        return list(labels)
    names = [f"LABEL_{i}" for i in range(num_classes)]
    names[predicted_index] = predicted_label
    if predicted_label != injection_label and num_classes == 2:
        # Two-class head where the safe class won: the other index must be
        # the injection class.
        names[1 - predicted_index] = injection_label
    return names
