from collections.abc import Sequence

from any_guardrail.types import AnyDict, CategoryResult, GuardrailInferenceOutput, GuardrailOutput, SpanResult


def normalize_rubric_to_risk(raw: float, lo: int, hi: int, *, higher_is_better: bool) -> float | None:
    """Map a rubric/likert score onto the canonical risk scale ~[0, 1] (higher = riskier).

    ``q = (raw - lo) / (hi - lo)`` is the normalized quality in [0, 1]; risk is its complement
    when higher rubric values are better, otherwise ``q`` directly. Returns None when the bounds
    are degenerate (``hi == lo``), since no scale can be inferred.
    """
    if hi == lo:
        return None
    quality = (raw - lo) / (hi - lo)
    quality = max(0.0, min(1.0, quality))  # clamp out-of-range rubric values
    return (1.0 - quality) if higher_is_better else quality


def _entity_type(label: str) -> str | None:
    """Strip a BIO/BIOES tag prefix, returning the entity type or None for an 'outside' tag."""
    if label in ("O", "o", ""):
        return None
    for prefix in ("B-", "I-", "E-", "S-", "L-", "U-"):
        if label.startswith(prefix):
            return label[len(prefix) :]
    return label


def spans_from_token_labels(
    label_ids: Sequence[int],
    offsets: Sequence[Sequence[int]],
    id2label: dict[int, str],
    text: str,
    scores: Sequence[float] | None = None,
) -> list[SpanResult]:
    """Merge per-token entity predictions into character-level ``SpanResult`` objects.

    Greedy decoding: contiguous tokens sharing an entity type (after stripping any
    ``B-``/``I-``/``E-``/``S-`` tag prefix) collapse into one span. Tokens tagged
    ``O``/outside and special tokens (offset ``(0, 0)``) are skipped. Robust to BIO,
    BIOES, IOB2, and entity-per-token schemes; Viterbi/CRF decoding is not applied.

    Args:
        label_ids: Predicted label id per token (argmax over the label dimension).
        offsets: ``(start, end)`` character offsets per token (from a fast tokenizer's
            ``return_offsets_mapping``). Special tokens carry ``(0, 0)``.
        id2label: Mapping from label id to label string.
        text: The validated text the offsets index into.
        scores: Optional per-token probability of the predicted label; averaged across
            a span to populate ``SpanResult.score``.

    Returns:
        Character spans for every detected entity, in order of appearance.

    """
    spans: list[SpanResult] = []
    current_type: str | None = None
    current_start = 0
    current_end = 0
    current_scores: list[float] = []

    def _close() -> None:
        nonlocal current_type, current_scores
        if current_type is not None:
            span_score = sum(current_scores) / len(current_scores) if current_scores else None
            spans.append(
                SpanResult(
                    start=current_start,
                    end=current_end,
                    text=text[current_start:current_end],
                    label=current_type,
                    score=span_score,
                )
            )
        current_type = None
        current_scores = []

    for index, (label_id, offset) in enumerate(zip(label_ids, offsets, strict=True)):
        off_start, off_end = int(offset[0]), int(offset[1])
        entity = _entity_type(id2label.get(int(label_id), "O")) if off_start != off_end else None
        token_score = float(scores[index]) if scores is not None else None
        if entity is not None and entity == current_type:
            current_end = off_end
            if token_score is not None:
                current_scores.append(token_score)
            continue
        _close()
        if entity is not None:
            current_type, current_start, current_end = entity, off_start, off_end
            current_scores = [token_score] if token_score is not None else []
    _close()
    return spans


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
    # The argmax decides the verdict for the whole row, so every class has a
    # known triggered state: the predicted class fired, the rest did not.
    categories = [
        CategoryResult(name=name, score=probability, triggered=name == predicted_label)
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
