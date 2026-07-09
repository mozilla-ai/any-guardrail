import os
import time
from typing import Any, ClassVar

import requests

from any_guardrail.base import Guardrail, GuardrailOutput
from any_guardrail.types import AnyDict, CategoryResult

# Lakera Guard v2 reports detection confidence as an ordinal *level*, not a
# probability. We map each level to a float in [0, 1] so ``GuardrailOutput.score``
# stays a single comparable number; the raw levels remain available per-detector
# in ``result.categories`` and ``extra["breakdown"]``.
# https://docs.lakera.ai/api-reference/lakera-api/guard/screen-content
_CONFIDENCE_SCORES: dict[str, float] = {
    "l1_confident": 1.0,
    "l2_very_likely": 0.8,
    "l3_likely": 0.6,
    "l4_less_likely": 0.4,
    "l5_unlikely": 0.2,
    "no_level": 0.0,
}


class LakeraGuard(Guardrail):
    """Lakera Guard — hosted API for prompt-injection, jailbreak, content-moderation, and PII detection (Lakera).

    Lakera Guard exposes a single ``/v2/guard`` endpoint that returns whether a message (or message list)
    was flagged. ``validate(content)`` accepts either a plain string (wrapped as a single user-role
    message) or a pre-formed chat-message list (``[{"role": "user", "content": "..."}]``), so both
    prompts and full conversations (including assistant turns) can be screened. By default this
    guardrail also opts into the endpoint's richer outputs so callers get the full picture of *why*
    something was flagged:

    - ``breakdown`` (requested via ``breakdown=True``): one entry per detector the policy ran, with its
      ``detector_type``, whether it ``detected`` a threat, and an ordinal confidence ``result``
      (``l1_confident`` … ``l5_unlikely`` / ``no_level``).
    - ``payload`` (requested via ``payload=True``): the string location (``start`` / ``end``), matched
      ``text``, ``detector_type``, and ``labels`` of any PII, profanity, or custom-regex matches.

    Auth is via a bearer token; obtain an API key from https://platform.lakera.ai/ (free Community tier:
    10k requests/month) and set it via the ``LAKERA_API_KEY`` environment variable or pass it directly.

    ``GuardrailOutput`` mapping:
        - ``valid = not flagged``.
        - ``score`` is the highest detector confidence among *detected* threats, mapped from the ordinal
          level to a float (``l1_confident`` → ``1.0`` … ``l5_unlikely`` → ``0.2``, higher = riskier);
          ``0.0`` when nothing was detected. If ``breakdown`` is disabled, ``score`` falls back to
          ``1.0`` when flagged else ``0.0``.
        - ``categories`` lists one ``CategoryResult`` per ``breakdown`` entry (``name`` =
          ``detector_type``, ``triggered`` = whether it detected, ``score`` = the mapped confidence).
        - ``extra`` carries the ``flagged`` flag, the ``payload`` list, the request ``metadata``
          (``request_uuid``), the convenience ``detected_detector_types`` list, and ``dev_info``
          when requested.
        - ``raw`` is the full Lakera response body (including the per-detector ``breakdown``).

    Research backing:
        - Pfister et al., *Gandalf the Red: Adaptive Security for LLMs*
          (https://arxiv.org/abs/2501.07927, 2025) introduces the D-SEC threat model and releases the
          279k-prompt Gandalf attack dataset that informs Lakera's training pipeline.
        - The differentiator vs. OSS DeBERTa-based prompt-injection detectors is the proprietary
          Gandalf-derived training data: 1M+ players and 80M+ adversarial prompts collected via
          Lakera's public Gandalf challenge platform. The Gandalf paper shows OSS detectors
          underperform on adaptive attacks at scale.

    For more information, see:

    - [Lakera platform (API keys, free Community tier)](https://platform.lakera.ai/)
    - [Product overview: prompt defense](https://www.lakera.ai/prompt-defense)
    - [API docs: Guard](https://docs.lakera.ai/docs/api/guard)
    - [API reference: screen content (`/v2/guard`)](https://docs.lakera.ai/api-reference/lakera-api/guard/screen-content)
    - [Gandalf the Red: Adaptive Security for LLMs (arXiv:2501.07927)](https://arxiv.org/abs/2501.07927)

    Brand transition note:
        Lakera was acquired by Cisco in 2025 and is being folded into Cisco AI Defense. The
        ``docs.lakera.ai`` API remains the public surface for now, with Pro/Enterprise tiers
        sales-gated through Cisco. Endpoint consolidation under Cisco AI Defense is expected within
        12-18 months; expect the constructor's ``endpoint`` default to be revised at that point.

    Args:
        api_key (str | None): The API key for authenticating with the Lakera Guard API. If not
            provided, it will be read from the ``LAKERA_API_KEY`` environment variable.
        endpoint (str): The Lakera Guard API endpoint URL. Defaults to the v2 endpoint at
            ``https://api.lakera.ai/v2/guard``.
        project_id (str | None): Optional Lakera project ID. Lakera projects allow per-project
            policy configuration (which detectors to run, severity thresholds, custom rules); when
            supplied, the project ID is forwarded with each request so the project's policy is applied.
        breakdown (bool): Request the per-detector ``breakdown`` list. Defaults to ``True``.
        payload (bool): Request the ``payload`` list locating PII / profanity / custom-regex matches.
            Defaults to ``True``.
        dev_info (bool): Request Lakera build information (git revision, model version) in the response.
            Defaults to ``False``.
        metadata (dict[str, Any] | None): Optional request metadata forwarded to Lakera for
            observability (e.g. ``user_id``, ``session_id``, ``ip_address``, ``internal_request_id``).

    """

    SUPPORTED_MODELS: ClassVar = ["lakera-guard"]

    def __init__(
        self,
        api_key: str | None = None,
        endpoint: str = "https://api.lakera.ai/v2/guard",
        project_id: str | None = None,
        breakdown: bool = True,
        payload: bool = True,
        dev_info: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the Lakera Guard guardrail with the provided configuration.

        Does not perform any network I/O — the API is only contacted when ``validate()`` is called.

        Args:
            api_key: API key for authenticating with the Lakera Guard API. If not provided,
                it is read from the ``LAKERA_API_KEY`` environment variable. Obtain one at
                https://platform.lakera.ai/ (free Community tier: 10k requests/month).
            endpoint: Lakera Guard API endpoint URL. Defaults to the v2 endpoint at
                ``https://api.lakera.ai/v2/guard``; override for self-hosted or regional
                deployments.
            project_id: Optional Lakera project ID (e.g. ``"project-1234"``). Projects carry
                per-project policy configuration (which detectors run, severity thresholds,
                custom rules); when supplied, it is forwarded with each request so that
                project's policy is applied.
            breakdown: If ``True`` (default), request the per-detector ``breakdown`` list, which
                also enables the graded ``score`` / ``categories`` mapping. If ``False``,
                ``score`` degrades to ``1.0``/``0.0`` and ``categories`` is empty.
            payload: If ``True`` (default), request the ``payload`` list locating PII /
                profanity / custom-regex matches (``start`` / ``end`` offsets, matched ``text``,
                ``labels``), surfaced in ``extra["payload"]``.
            dev_info: If ``True``, request Lakera build information (git revision, model
                version) in the response, surfaced in ``extra["dev_info"]``. Defaults to
                ``False``.
            metadata: Optional request metadata forwarded to Lakera for observability, e.g.
                ``{"user_id": "u-42", "session_id": "s-1"}``.

        Raises:
            ValueError: If no API key is provided and ``LAKERA_API_KEY`` is not set.

        """
        if api_key:
            self.api_key = api_key
        elif os.getenv("LAKERA_API_KEY"):
            self.api_key = os.getenv("LAKERA_API_KEY")  # type: ignore[assignment]
        else:
            msg = (
                "API key must be provided either as the `api_key=` parameter or through the "
                "LAKERA_API_KEY environment variable. Sign up at https://platform.lakera.ai/ to "
                "obtain a key (free Community tier available)."
            )
            raise ValueError(msg)

        self.endpoint = endpoint
        self.project_id = project_id
        self.breakdown = breakdown
        self.payload = payload
        self.dev_info = dev_info
        self.metadata = metadata

    def validate(
        self,
        content: str | list[dict[str, str]],
    ) -> GuardrailOutput:
        """Validate a string or chat-message list against the Lakera Guard API.

        Args:
            content (str | list[dict[str, str]]): Either a plain string (wrapped as a single
                user-role message, the common prompt-screening case) or a pre-formed list of
                chat messages following the ``[{"role": "user", "content": "..."}]`` shape.
                Pass the message-list form to screen a whole conversation, e.g.
                ``[{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "..."}]``.

        Returns:
            ``GuardrailOutput`` with ``valid = not flagged``, ``score`` derived from the highest
            detected-detector confidence level (higher = riskier), ``categories`` (one per
            ``breakdown`` entry), and ``extra`` carrying the ``flagged`` flag, ``payload``,
            ``metadata``, ``detected_detector_types``, and (when requested) ``dev_info``.
            ``raw`` holds the full response body (including the per-detector ``breakdown``).
            ``usage.latency_ms`` records the round-trip time.

        Raises:
            ValueError: If ``content`` is neither a string nor a list of message dicts, or if
                the API responds with a non-200 status code.

        """
        start = time.perf_counter()
        params = self._pre_processing(content)
        response = self._inference(params)
        result = self._post_processing(response)
        self._stamp_usage(result, (time.perf_counter() - start) * 1000.0)
        return result

    def _pre_processing(self, content: str | list[dict[str, str]]) -> AnyDict:
        if isinstance(content, str):
            messages = [{"role": "user", "content": content}]
        elif isinstance(content, list):
            messages = content
        else:
            msg = "Content must be either a string or a list of message dictionaries."
            raise ValueError(msg)

        body: AnyDict = {
            "messages": messages,
            "breakdown": self.breakdown,
            "payload": self.payload,
        }
        if self.dev_info:
            body["dev_info"] = True
        if self.project_id:
            body["project_id"] = self.project_id
        if self.metadata:
            body["metadata"] = self.metadata
        return body

    def _inference(self, params: AnyDict) -> requests.Response:
        response = requests.post(
            self.endpoint,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=params,
        )
        if response.status_code != 200:
            msg = f"Request to Lakera Guard API failed with status code {response.status_code}: {response.text}"
            raise ValueError(msg)
        return response

    def _post_processing(
        self,
        response: requests.Response,
    ) -> GuardrailOutput:
        body = response.json()
        flagged = bool(body.get("flagged", False))
        breakdown = body.get("breakdown") or []
        payload = body.get("payload") or []
        metadata = body.get("metadata") or {}

        detected = [entry for entry in breakdown if entry.get("detected")]
        if detected:
            score = max(_CONFIDENCE_SCORES.get(entry.get("result"), 0.0) for entry in detected)
        elif flagged:
            # Flagged but the per-detector breakdown wasn't requested; no level to map.
            score = 1.0
        else:
            score = 0.0

        categories = [
            CategoryResult(
                name=entry.get("detector_type") or "unknown",
                triggered=bool(entry.get("detected")),
                score=_CONFIDENCE_SCORES.get(entry.get("result"), 0.0),
            )
            for entry in breakdown
        ]

        extra: dict[str, Any] = {
            "flagged": flagged,
            "payload": payload,
            "metadata": metadata,
            "detected_detector_types": sorted(
                {entry.get("detector_type") for entry in detected if entry.get("detector_type")}
            ),
        }
        dev_info = body.get("dev_info")
        if dev_info:
            extra["dev_info"] = dev_info

        return GuardrailOutput(
            valid=not flagged,
            score=score,
            categories=categories,
            extra=extra,
            raw=body,
        )
