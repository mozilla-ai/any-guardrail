import math
import time
from typing import ClassVar
from urllib.parse import urlsplit

import requests

from any_guardrail.base import Guardrail, GuardrailName
from any_guardrail.registry import GUARDRAIL_METADATA
from any_guardrail.taxonomy import GuardrailMetadata
from any_guardrail.types import GuardrailOutput


class Deepkeep(Guardrail):
    """DeepKeep firewall API for input screening with configurable safety policies.

    Uses ``POST /api/v3/openai/moderations/pre`` with ``X-API-Key`` authentication.
    The firewall ID is sent as ``model``; it is not the inference model name.
    No network I/O occurs during construction and no model weights are loaded.

    This initial adapter is validate-only: ``valid = not flagged``, including
    flagged alert, redact, and modify outcomes. It never rewrites input or treats
    a recommended action as permission to forward flagged, unmodified text.
    Configure a detection/block firewall for this integration. A missing boolean
    verdict or an API failure raises rather than returning a safe verdict, so a
    gateway can apply its own unavailable policy. No calibrated score is invented.

    ``raw`` contains the backend response and may contain sensitive text. Logging
    at DeepKeep is explicitly disabled by default; this does not replace reviewing
    the deployment's retention policy. This adapter supports input text only,
    not output inspection, streaming, images, or batched requests.

    Args:
        api_key: DeepKeep API key, sent in the X-API-Key header. Required.
        endpoint: Deployment base URL, without the moderation path. Required.
        firewall_id: Input firewall ID sent as the model field. Required.
        timeout: Positive finite HTTP connect/read timeout in seconds, not a total deadline.
        logs: Whether to request DeepKeep content logging. Defaults to False.

    """

    SUPPORTED_MODELS: ClassVar[list[str]] = []
    METADATA: ClassVar[GuardrailMetadata] = GUARDRAIL_METADATA[GuardrailName.DEEPKEEP]

    def __init__(
        self,
        api_key: str,
        endpoint: str,
        firewall_id: str,
        timeout: float = 10.0,
        logs: bool = False,
    ) -> None:
        """Validate and store deployment configuration without contacting DeepKeep.

        Args:
            api_key: DeepKeep API key. Required; stored as a secret by Otari.
            endpoint: HTTP(S) deployment base URL; use HTTPS outside trusted local networks.
            firewall_id: DeepKeep input firewall ID, not the LLM model name.
            timeout: Positive finite connect/read timeout in seconds. Defaults to 10.
            logs: Request content logging at DeepKeep. Defaults to False.

        Raises:
            ValueError: If required configuration is empty or invalid.

        """
        if not isinstance(api_key, str) or not api_key.strip():
            msg = "A DeepKeep api_key is required."
            raise ValueError(msg)
        if not isinstance(firewall_id, str) or not firewall_id.strip():
            msg = "A DeepKeep firewall_id is required."
            raise ValueError(msg)
        try:
            url = urlsplit(endpoint)
            valid_url = (
                url.scheme in {"http", "https"}
                and bool(url.hostname)
                and url.username is None
                and url.password is None
                and not url.query
                and not url.fragment
            )
            _ = url.port
        except (TypeError, ValueError, AttributeError):
            valid_url = False
        if not valid_url:
            msg = "DeepKeep endpoint must be an HTTP(S) base URL without credentials, query, or fragment."
            raise ValueError(msg)
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, int | float)
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            msg = "DeepKeep timeout must be a positive finite number."
            raise ValueError(msg)
        if not isinstance(logs, bool):
            msg = "DeepKeep logs must be a boolean."
            raise ValueError(msg)
        self.api_key = api_key
        self.endpoint = endpoint.rstrip("/")
        self.firewall_id = firewall_id
        self.timeout = timeout
        self.logs = logs

    def validate(self, input_text: str) -> GuardrailOutput:
        """Screen text and return a binary verdict, with the backend response in raw.

        Args:
            input_text: Plain input text to screen against the configured firewall.

        Returns:
            GuardrailOutput with valid equal to not flagged, no risk score, and latency.

        Raises:
            TypeError: If input_text is not a string.
            ValueError: If the API fails, redirects, returns invalid JSON, or omits a boolean verdict.

        """
        if not isinstance(input_text, str):
            msg = "DeepKeep input_text must be a string."
            raise TypeError(msg)
        start = time.perf_counter()
        try:
            response = requests.post(
                f"{self.endpoint}/api/v3/openai/moderations/pre",
                headers={"X-API-Key": self.api_key},
                json={"model": self.firewall_id, "input": input_text, "logs": self.logs},
                timeout=self.timeout,
                allow_redirects=False,
            )
        except requests.RequestException:
            msg = "DeepKeep moderation request failed."
            raise ValueError(msg) from None
        if not 200 <= response.status_code < 300:
            msg = f"DeepKeep moderation returned HTTP {response.status_code}."
            raise ValueError(msg)
        try:
            body = response.json()
        except ValueError:
            msg = "DeepKeep moderation returned invalid JSON."
            raise ValueError(msg) from None
        if not isinstance(body, dict) or not isinstance(body.get("flagged"), bool):
            msg = "DeepKeep moderation response must contain a boolean flagged verdict."
            raise ValueError(msg)
        result = GuardrailOutput(valid=not body["flagged"], raw=body)
        self._stamp_usage(result, (time.perf_counter() - start) * 1000.0)
        return result
