import os
import time
from typing import Any, ClassVar

import requests

from any_guardrail.base import Guardrail, GuardrailOutput
from any_guardrail.types import AnyDict, CategoryResult

# GLiGuard's documented gliner2 classification schema (single binary task).
_DEFAULT_SCHEMA: AnyDict = {"prompt_safety": ["safe", "unsafe"]}

# Predicted labels that count as *passing* (not a safety violation). Any other
# predicted label flags the task. Covers GLiGuard's binary-safety, toxicity, and
# refusal label sets; override via the ``safe_labels`` constructor arg.
_DEFAULT_SAFE_LABELS = frozenset({"safe", "benign", "compliance", "refusal"})

# Envelope keys Pioneer might wrap the gliner2 ``{task: label}`` output in. The
# native /inference response body is not publicly documented, so post-processing
# unwraps any of these (or uses the body itself) before reading predictions.
_ENVELOPE_KEYS = ("predictions", "prediction", "result", "results", "output", "classifications", "data")


def _to_labels(value: Any) -> list[str]:
    """Normalize a gliner2 task prediction to a flat list of label strings."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        # e.g. {"label": "unsafe", "score": 0.9}
        if "label" in value and isinstance(value["label"], str):
            return [value["label"]]
        # e.g. {"safe": 0.05, "unsafe": 0.95} -> the argmax label, not every key
        if value and all(isinstance(v, (int, float)) for v in value.values()):
            return [max(value, key=lambda k: value[k])]
        return [str(k) for k in value]
    if isinstance(value, list):
        labels: list[str] = []
        for item in value:
            if isinstance(item, str):
                labels.append(item)
            elif isinstance(item, dict) and isinstance(item.get("label"), str):
                labels.append(item["label"])
        return labels
    return []


class Pioneer(Guardrail):
    """Wraps Fastino's Pioneer inference API running the GLiGuard guardrail model.

    This is the hosted, pay-per-use counterpart to the open-weight GLiGuard model
    (``fastino/gliguard-LLMGuardrails-300M``): the same schema-conditioned safety
    classifier, served on Fastino's Pioneer platform so you can run content
    moderation / prompt-injection / jailbreak / harm classification in one forward
    pass without the local ``gliner2`` runtime dependency.

    A request sends the text plus a *classification schema* (one or more tasks,
    each with its candidate labels). The default schema runs GLiGuard's binary
    ``prompt_safety`` task (``"safe"`` / ``"unsafe"``); supply a richer schema to
    add toxicity-category or jailbreak-strategy tasks in the same call.

    Auth is via an API key. Obtain one from https://pioneer.ai/ (Hobby tier
    includes monthly inference credits) and set it via ``PIONEER_API_KEY`` or pass
    it directly.

    .. note::
        Pioneer's ``/inference`` *response* body is not publicly documented
        (the request, auth, model id, and pricing are). Post-processing therefore
        parses the response defensively: it unwraps a common envelope key if
        present, reads each task's predicted label(s), and treats any label not in
        ``safe_labels`` as a violation. The full response is always available in
        ``raw``. Confirm the mapping against a live key before relying on it in
        production.

    ``GuardrailOutput`` mapping:
        - ``valid = no task predicted a non-passing label``.
        - ``score`` is ``1.0`` when any task is flagged, else ``0.0`` (GLiGuard
          returns labels via this schema, not calibrated probabilities).
        - ``categories`` lists one ``CategoryResult`` per task (``name`` = task,
          ``description`` = predicted label(s), ``triggered`` = flagged).
        - ``extra`` carries the parsed ``predictions`` and ``model_id``; ``raw`` is
          the full response body.
        - Fails closed when no predictions can be read from the response.

    Research backing:
        - Zaratiana et al., *GLiGuard: Schema-Conditioned Classification for LLM
          Safeguard* (https://arxiv.org/abs/2605.07982, 2026).
        - Model card: https://huggingface.co/fastino/gliguard-LLMGuardrails-300M
        - Docs: https://docs.pioneer.ai/

    Args:
        api_key (str | None): Pioneer API key. Falls back to ``PIONEER_API_KEY``.
        model_id (str): The Pioneer model id. Defaults to
            ``fastino/gliguard-llm-guardrails-300m``.
        endpoint (str): Inference endpoint. Defaults to
            ``https://api.pioneer.ai/inference``.
        schema (dict | None): The gliner2 classification schema. Defaults to
            ``{"prompt_safety": ["safe", "unsafe"]}``.
        threshold (float): Confidence threshold forwarded to the model. Defaults
            to ``0.5``.
        safe_labels (set[str] | None): Predicted labels that count as passing.
            Defaults to ``{"safe", "benign", "compliance", "refusal"}``.

    """

    SUPPORTED_MODELS: ClassVar = ["fastino/gliguard-llm-guardrails-300m"]

    def __init__(
        self,
        api_key: str | None = None,
        model_id: str = "fastino/gliguard-llm-guardrails-300m",
        endpoint: str = "https://api.pioneer.ai/inference",
        schema: AnyDict | None = None,
        threshold: float = 0.5,
        safe_labels: set[str] | None = None,
    ) -> None:
        """Initialize the Pioneer guardrail.

        Does not perform any network I/O — the API is only contacted on
        ``validate()``.
        """
        if api_key:
            self.api_key = api_key
        elif os.getenv("PIONEER_API_KEY"):
            self.api_key = os.getenv("PIONEER_API_KEY")  # type: ignore[assignment]
        else:
            msg = (
                "API key must be provided either as the `api_key=` parameter or through the "
                "PIONEER_API_KEY environment variable. Sign up at https://pioneer.ai/ to obtain a key."
            )
            raise ValueError(msg)

        self.model_id = model_id
        self.endpoint = endpoint
        self.schema = schema if schema is not None else dict(_DEFAULT_SCHEMA)
        self.threshold = threshold
        self.safe_labels = safe_labels if safe_labels is not None else set(_DEFAULT_SAFE_LABELS)

    def validate(self, content: str) -> GuardrailOutput:
        """Classify ``content`` with the configured GLiGuard schema.

        Args:
            content (str): The text to classify.

        Returns:
            ``GuardrailOutput`` with ``valid = no task flagged`` (see the class
            docstring for the field mapping). Fails closed (``valid=False`` with
            ``extra={"parse_failure": True}``) when no predictions are found.

        """
        start = time.perf_counter()
        params = self._pre_processing(content)
        response = self._inference(params)
        result = self._post_processing(response)
        self._stamp_usage(result, (time.perf_counter() - start) * 1000.0)
        return result

    def _pre_processing(self, content: str) -> AnyDict:
        return {
            "model_id": self.model_id,
            "text": content,
            "schema": self.schema,
            "threshold": self.threshold,
        }

    def _inference(self, params: AnyDict) -> requests.Response:
        response = requests.post(
            self.endpoint,
            headers={"X-API-Key": self.api_key},
            json=params,
        )
        if response.status_code != 200:
            msg = f"Request to Pioneer API failed with status code {response.status_code}: {response.text}"
            raise ValueError(msg)
        return response

    def _post_processing(self, response: requests.Response) -> GuardrailOutput:
        body = response.json()
        predictions = self._extract_predictions(body)
        if not predictions:
            return GuardrailOutput(valid=False, extra={"parse_failure": True}, raw=body)

        categories: list[CategoryResult] = []
        any_flagged = False
        for task, value in predictions.items():
            labels = _to_labels(value)
            flagged = any(label not in self.safe_labels for label in labels) if labels else False
            any_flagged = any_flagged or flagged
            categories.append(
                CategoryResult(
                    name=str(task),
                    description=", ".join(labels) if labels else None,
                    triggered=flagged,
                )
            )

        return GuardrailOutput(
            valid=not any_flagged,
            score=1.0 if any_flagged else 0.0,
            categories=categories,
            extra={"predictions": predictions, "model_id": self.model_id},
            raw=body,
        )

    @staticmethod
    def _extract_predictions(body: Any) -> AnyDict | None:
        """Find the ``{task: label}`` predictions dict in a Pioneer response body.

        The response envelope is undocumented, so this unwraps a recognized
        wrapper key (dict- or list-valued) when present, otherwise treats the body
        itself as the predictions mapping. Returns ``None`` (so the caller fails
        closed) when no task predictions can be recovered — never a phantom
        verdict named after a wrapper/metadata key.
        """
        if not isinstance(body, dict):
            return None
        for key in _ENVELOPE_KEYS:
            inner = body.get(key)
            if isinstance(inner, dict) and inner:
                return inner
            if isinstance(inner, list) and inner:
                merged: AnyDict = {}
                for element in inner:
                    if isinstance(element, dict):
                        merged.update(element)
                if merged:
                    return merged
        # No recognized envelope: treat the body itself as the predictions dict,
        # dropping known metadata and (now-empty) envelope keys so an empty
        # ``{"predictions": {}}`` fails closed instead of becoming a phantom task.
        skip = {"inference_id", "model_id", "id", "usage"} | set(_ENVELOPE_KEYS)
        candidate = {k: v for k, v in body.items() if k not in skip}
        return candidate or None
