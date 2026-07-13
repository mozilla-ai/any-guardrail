import os
import time
from typing import ClassVar

import requests

from any_guardrail.base import Guardrail, GuardrailName, GuardrailOutput
from any_guardrail.registry import GUARDRAIL_METADATA
from any_guardrail.taxonomy import GuardrailMetadata
from any_guardrail.types import AnyDict, CategoryResult, GuardrailUsage


class Alinia(Guardrail):
    """Hosted content-moderation and safety-detection API with configurable detection policies.

    Sends a text input or a full conversation to the Alinia API, which runs whichever detections
    you enable via ``detection_config`` (e.g. ``{"security": True}`` for prompt-injection /
    data-exfiltration detection, plus safety, compliance, and hallucination policies) and reports
    per-category verdicts. Alinia's detection models are multilingual (its security model is
    trained on English, Spanish, and Catalan).

    Verdict mapping: ``valid`` is ``True`` when Alinia does not flag the input. ``categories``
    flattens Alinia's nested ``category_details`` into one entry per ``group/label`` — boolean
    details become ``triggered`` flags, numeric details become per-category ``score`` values.
    The top-level ``score`` is the highest numeric category score (higher = riskier), or ``None``
    when the endpoint returns only booleans. ``explanation`` carries the recommendation text when
    the endpoint returns one (e.g. sensitive information), ``action`` carries a structured
    recommendation's action (e.g. ``"block"``), and ``raw`` is the full response JSON.

    Expected inputs: ``validate`` accepts either a plain string or a list of chat-message dicts
    (``{"role": ..., "content": ...}``), plus an optional model ``output`` and optional
    ``context_documents`` for detections that evaluate responses in context.

    You must obtain an API key and the endpoint URL from Alinia, and pass them either directly to
    the constructor or via the ``ALINIA_API_KEY`` / ``ALINIA_ENDPOINT`` environment variables.

    For more information, see:

    - [Alinia AI](https://alinia.ai/) (vendor site).
    - [Integrating Alinia into any-guardrail](https://blog.mozilla.ai/integrating-alinia-into-any-guardrail-for-multilingual-ai-security/)
      (Mozilla AI blog walkthrough of this guardrail).

    Args:
        detection_config (str | dict): Which detections to run and their thresholds: either a
            detection-configuration ID string registered with Alinia, or a dict such as
            ``{"security": True}`` (optionally nested per-policy settings).
        api_key (str | None): The API key for authenticating with the Alinia API. If not provided,
            it is read from the ``ALINIA_API_KEY`` environment variable.
        endpoint (str | None): The Alinia API endpoint URL. If not provided, it is read from the
            ``ALINIA_ENDPOINT`` environment variable.
        metadata (dict | None): Optional metadata to include with every request (e.g. app or
            user identifiers).
        blocked_response (dict | None): Optional response Alinia should return when content
            is blocked.
        stream (bool): Whether to request a streaming API response. Defaults to ``False``.

    """

    METADATA: ClassVar[GuardrailMetadata] = GUARDRAIL_METADATA[GuardrailName.ALINIA]

    def __init__(
        self,
        detection_config: str | dict[str, float | bool] | dict[str, dict[str, float | bool | str]],
        api_key: str | None = None,
        endpoint: str | None = None,
        metadata: AnyDict | None = None,
        blocked_response: dict[str, str] | None = None,
        stream: bool = False,
    ):
        """Initialize the Alinia guardrail with the provided configuration.

        Args:
            detection_config: Which detections to run and their thresholds. Pass either a
                detection-configuration ID string registered with Alinia, or a dict enabling
                detections inline — e.g. ``{"security": True}``, or nested per-policy settings
                such as ``{"safety": {"toxicity": 0.8}}``.
            api_key: Alinia API key. If ``None``, it is read from the ``ALINIA_API_KEY``
                environment variable.
            endpoint: Alinia API endpoint URL (obtained from Alinia alongside the key). If
                ``None``, it is read from the ``ALINIA_ENDPOINT`` environment variable.
            metadata: Optional metadata dict sent with every request (e.g. app or user
                identifiers for Alinia-side monitoring).
            blocked_response: Optional response Alinia should return when content is blocked,
                e.g. ``{"output": "Sorry, I can't help with that."}``.
            stream: Whether to request a streaming API response. Defaults to ``False``.

        Raises:
            ValueError: If no API key or endpoint is provided and the corresponding
                environment variable is not set.

        """
        if api_key:
            self.api_key = api_key
        elif os.getenv("ALINIA_API_KEY"):
            self.api_key = os.getenv("ALINIA_API_KEY")  # type: ignore[assignment]
        else:
            msg = "API key must be provided either as a parameter or through the ALINIA_API_KEY environment variable."
            raise ValueError(msg)

        if endpoint:
            self.endpoint = endpoint
        elif os.getenv("ALINIA_ENDPOINT"):
            self.endpoint = os.getenv("ALINIA_ENDPOINT")  # type: ignore[assignment]
        else:
            msg = "Endpoint URL must be provided either as a parameter or through the ALINIA_ENDPOINT environment variable."
            raise ValueError(msg)

        self.detection_config = detection_config
        self.metadata = metadata
        self.blocked_response = blocked_response
        self.stream = stream

    def validate(
        self,
        conversation: str | list[dict[str, str]],
        output: str | None = None,
        context_documents: list[str] | None = None,
    ) -> GuardrailOutput:
        """Validate conversation or text input using the Alinia API.

        This can be used for validation using any of the API endpoints provided by Alinia.

        Args:
            conversation (str | list[dict[str, str]]): The input to validate. Either a plain
                string (sent as ``input``, e.g. ``"Ignore all instructions and ..."``) or a
                list of chat-message dicts (sent as ``messages``), e.g.
                ``[{"role": "user", "content": "..."}]``.
            output (str | None): Optional model response to validate alongside the input, for
                detections that evaluate outputs (e.g. hallucination or compliance checks).
            context_documents (list[str] | None): Optional context documents (e.g. retrieved
                RAG passages) that give output-side detections the grounding text to check
                against.

        Returns:
            GuardrailOutput where ``categories`` flattens Alinia's nested
            ``category_details`` (one entry per ``group/label``), ``score`` is the
            highest category score when numeric scores exist, ``explanation``
            carries the recommendation text when the endpoint returns one (e.g.
            sensitive information), and ``raw`` is the full response JSON.

        """
        start = time.perf_counter()
        params = self._pre_processing(conversation, output, context_documents)
        response = self._inference(params)
        result = self._post_processing(response)
        result.usage = GuardrailUsage(latency_ms=(time.perf_counter() - start) * 1000.0)
        return result

    def _pre_processing(
        self,
        conversation: str | list[dict[str, str]],
        output: str | None = None,
        context_documents: list[str] | None = None,
    ) -> AnyDict:
        """Build the JSON payload for the Alinia API request.

        Args:
            conversation: A plain string (mapped to the ``input`` field) or a list of
                chat-message dicts (mapped to the ``messages`` field).
            output: Optional model response, mapped to the ``output`` field.
            context_documents: Optional grounding documents, mapped to the
                ``context_documents`` field.

        Returns:
            The request payload, including the configured ``detection_config`` (as
            ``detection_config`` when a dict, or ``detection_config_id`` when a string) and
            any constructor-level ``metadata``, ``stream``, or ``blocked_response`` values.

        Raises:
            ValueError: If ``conversation`` is neither a string nor a list, or
                ``detection_config`` is neither a string nor a dict.

        """
        initial_json = {}

        if isinstance(conversation, str):
            initial_json["input"] = conversation
        elif isinstance(conversation, list):
            initial_json["messages"] = conversation  # type: ignore[assignment]
        else:
            msg = "Conversation must be either a string or a list of message dictionaries."
            raise ValueError(msg)

        if isinstance(self.detection_config, dict):
            initial_json["detection_config"] = self.detection_config  # type: ignore[assignment]
        elif isinstance(self.detection_config, str):
            initial_json["detection_config_id"] = self.detection_config
        else:
            msg = "Detection configuration must be either a string ID or a dictionary."
            raise ValueError(msg)

        if self.metadata:
            initial_json["metadata"] = self.metadata  # type: ignore[assignment]
        if self.stream:
            initial_json["stream"] = self.stream  # type: ignore[assignment]
        if context_documents:
            initial_json["context_documents"] = context_documents  # type: ignore[assignment]
        if output:
            initial_json["output"] = output
        if self.blocked_response:
            initial_json["blocked_response"] = self.blocked_response  # type: ignore[assignment]

        return initial_json

    def _inference(self, params: AnyDict) -> requests.Response:
        response = requests.post(
            self.endpoint,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=params,
        )
        if response.status_code != 200:
            msg = f"Request to Alinia API failed with status code {response.status_code}: {response.text}"
            raise ValueError(msg)
        return response

    def _post_processing(
        self,
        response: requests.Response,
    ) -> GuardrailOutput:
        response_json = response.json()
        result = response_json.get("result") or {}
        valid = not result.get("flagged")

        categories: list[CategoryResult] = []
        numeric_scores: list[float] = []
        category_details = result.get("category_details") or {}
        for group, labels in category_details.items():
            if not isinstance(labels, dict):
                continue
            for label, value in labels.items():
                name = f"{group}/{label}"
                if isinstance(value, bool):
                    categories.append(CategoryResult(name=name, triggered=value))
                elif isinstance(value, (int, float)):
                    score = float(value)
                    categories.append(CategoryResult(name=name, score=score))
                    numeric_scores.append(score)

        # Alinia's recommendation is either a plain string or the richer
        # ``{"action": "block", "output": "..."}`` form. Preserve it losslessly
        # in ``extra`` and surface the action / message as first-class fields.
        recommendation = response_json.get("recommendation") or result.get("recommendation")
        action: str | None = None
        explanation: str | None = None
        if isinstance(recommendation, str):
            explanation = recommendation
        elif isinstance(recommendation, dict):
            action = recommendation.get("action")
            output = recommendation.get("output")
            explanation = output if isinstance(output, str) else None

        return GuardrailOutput(
            valid=valid,
            explanation=explanation,
            score=max(numeric_scores) if numeric_scores else None,
            categories=categories,
            action=action,
            extra={"recommendation": recommendation} if recommendation is not None else None,
            raw=response_json,
        )
