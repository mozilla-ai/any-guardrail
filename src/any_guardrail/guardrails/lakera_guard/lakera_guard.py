import os
from typing import Any, ClassVar

import requests

from any_guardrail.base import Guardrail, GuardrailOutput
from any_guardrail.types import AnyDict


class LakeraGuard(Guardrail[bool, dict[str, Any], float]):
    """Wraps the Lakera Guard REST API for prompt-injection, jailbreak, content-moderation, and PII detection.

    Lakera Guard exposes a single ``/v2/guard`` endpoint that returns whether a message (or message list)
    was flagged, along with per-category booleans and scores. Auth is via a bearer token; you must
    obtain an API key from https://platform.lakera.ai/ (free Community tier: 10k requests/month) and
    set it via the ``LAKERA_API_KEY`` environment variable or pass it directly to the constructor.

    Research backing:
        - Pfister et al., *Gandalf the Red: Adaptive Security for LLMs*
          (https://arxiv.org/abs/2501.07927, 2025) introduces the D-SEC threat model and releases the
          279k-prompt Gandalf attack dataset that informs Lakera's training pipeline.
        - The differentiator vs. OSS DeBERTa-based prompt-injection detectors is the proprietary
          Gandalf-derived training data: 1M+ players and 80M+ adversarial prompts collected via
          Lakera's public Gandalf challenge platform. The Gandalf paper shows OSS detectors
          underperform on adaptive attacks at scale.
        - Product overview: https://www.lakera.ai/prompt-defense
        - API docs: https://docs.lakera.ai/guard

    Brand transition note:
        Lakera was acquired by Cisco in 2025 and is being folded into Cisco AI Defense. The
        ``docs.lakera.ai/guard`` API remains the public surface for now, with Pro/Enterprise tiers
        sales-gated through Cisco. Endpoint consolidation under Cisco AI Defense is expected within
        12-18 months; expect the constructor's ``endpoint`` default to be revised at that point.

    Args:
        api_key (str | None): The API key for authenticating with the Lakera Guard API. If not
            provided, it will be read from the ``LAKERA_API_KEY`` environment variable.
        endpoint (str): The Lakera Guard API endpoint URL. Defaults to the v2 endpoint at
            ``https://api.lakera.ai/v2/guard``.
        project_id (str | None): Optional Lakera project ID. Lakera projects allow per-project
            policy configuration (which categories to flag, severity thresholds, custom rules); when
            supplied, the project ID is forwarded with each request so the project's policy is
            applied.

    """

    SUPPORTED_MODELS: ClassVar = ["lakera-guard"]

    def __init__(
        self,
        api_key: str | None = None,
        endpoint: str = "https://api.lakera.ai/v2/guard",
        project_id: str | None = None,
    ) -> None:
        """Initialize the Lakera Guard guardrail with the provided configuration.

        Does not perform any network I/O — the API is only contacted when ``validate()`` is called.
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

    def validate(
        self,
        content: str | list[dict[str, str]],
    ) -> GuardrailOutput[bool, dict[str, Any], float]:
        """Validate a string or chat-message list against the Lakera Guard API.

        Args:
            content (str | list[dict[str, str]]): Either a plain string (wrapped as a single
                user-role message) or a pre-formed list of chat messages following the
                ``[{"role": "user", "content": "..."}]`` shape.

        Returns:
            ``GuardrailOutput`` with ``valid = not response["flagged"]``,
            ``score = max(category_scores.values())`` (or ``0.0`` when empty), and ``explanation``
            containing the raw ``categories``, ``category_scores``, and ``results`` from the API.

        """
        params = self._pre_processing(content)
        response = self._inference(params)
        return self._post_processing(response)

    def _pre_processing(self, content: str | list[dict[str, str]]) -> AnyDict:
        if isinstance(content, str):
            messages = [{"role": "user", "content": content}]
        elif isinstance(content, list):
            messages = content
        else:
            msg = "Content must be either a string or a list of message dictionaries."
            raise ValueError(msg)

        payload: AnyDict = {"messages": messages}
        if self.project_id:
            payload["project_id"] = self.project_id
        return payload

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
    ) -> GuardrailOutput[bool, dict[str, Any], float]:
        body = response.json()
        valid = not body.get("flagged", False)
        category_scores: dict[str, float] = body.get("category_scores", {}) or {}
        score = max(category_scores.values()) if category_scores else 0.0
        explanation: dict[str, Any] = {
            "categories": body.get("categories", {}),
            "category_scores": category_scores,
            "results": body.get("results", []),
        }
        return GuardrailOutput(
            valid=valid,
            explanation=explanation,
            score=score,
        )
