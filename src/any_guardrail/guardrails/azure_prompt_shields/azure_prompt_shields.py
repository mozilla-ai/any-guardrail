import os
import time
from typing import Any, ClassVar

from any_guardrail.base import GuardrailName
from any_guardrail.registry import GUARDRAIL_METADATA
from any_guardrail.taxonomy import GuardrailMetadata

try:
    import requests
    from azure.core.credentials import AzureKeyCredential
except ImportError as e:
    msg = (
        "Dependencies for AzurePromptShields are missing (`requests` and `azure-core`, "
        "which ship with the `azure-content-safety` extra). "
        "Please install them with `pip install 'any-guardrail[azure-content-safety]'`."
    )
    raise ImportError(msg) from e

from any_guardrail.base import Guardrail, GuardrailOutput
from any_guardrail.types import CategoryResult, GuardrailUsage

# Azure Prompt Shields REST API version (GA).
# See https://learn.microsoft.com/en-us/rest/api/contentsafety/text-operations/detect-text-jailbreak
_API_VERSION = "2024-09-01"


class AzurePromptShields(Guardrail):
    """Hosted detector for direct (user prompt) and indirect (document-borne) prompt-injection and jailbreak attacks.

    Prompt Shields is a service from Azure AI Content Safety that detects prompt-injection
    and jailbreak attacks against LLM applications. It supports two attack surfaces:

    - **Direct attacks (user_prompt)**: malicious instructions in end-user input attempting
      to override the system prompt, exfiltrate sensitive info, or otherwise jailbreak
      the model.
    - **Indirect attacks (documents)**: data-borne prompt injection embedded inside
      retrieved documents, tool outputs, or other context fed to the model. Microsoft
      Research's [Spotlighting](https://arxiv.org/abs/2403.14720) technique (Hines et al., 2024)
      is the published basis for this indirect-attack detection.

    Expected inputs: ``validate`` takes an optional ``user_prompt`` (a single string, the
    end-user prompt) and/or optional ``documents`` (a list of strings — retrieved context,
    tool outputs, etc.). At least one of the two must be provided; each surface is only
    analyzed when its argument is supplied.

    ``GuardrailOutput`` mapping:
        - ``valid`` is ``True`` iff Azure detects no attack anywhere — neither in the user
          prompt nor in **any** supplied document.
        - ``score`` is a binary severity proxy: ``1.0`` when any attack is detected, ``0.0``
          otherwise (higher = riskier). Prompt Shields returns per-surface booleans rather
          than a continuous risk probability.
        - ``categories`` holds one ``CategoryResult`` per analyzed source — ``user_prompt``
          (when supplied) and ``document_{i}`` for each document — with ``triggered`` set to
          that surface's ``attackDetected`` flag.
        - ``extra`` carries the per-field detection booleans; ``raw`` is the full REST
          response. A malformed / unparsable Azure payload fails closed (``valid=False``,
          ``score=1.0``, ``extra={"parse_failure": True}``).

    This guardrail hits the same Azure Content Safety resource as ``AzureContentSafety`` and
    reuses the same env vars (``CONTENT_SAFETY_KEY`` / ``CONTENT_SAFETY_ENDPOINT``) and the
    ``azure-content-safety`` optional extra. It is, however, a separate guardrail because the
    threat surface, request payload, and response shape differ from the content-harm endpoint.

    Implementation note:
        The current ``azure-ai-contentsafety`` Python SDK (``>=1.0.0``) does not yet expose a
        ``shield_prompt`` method on ``ContentSafetyClient``. This guardrail therefore calls the
        Prompt Shields REST endpoint directly via ``requests.post`` using the API version
        ``2024-09-01``. If a future SDK release adds first-class support, this class can be
        switched over without changing the public ``validate()`` signature.

    Caveat on real-world robustness:
        An independent evaluation
        ([arXiv:2504.11168](https://arxiv.org/pdf/2504.11168), 2025) reports large evasion gaps
        for hosted prompt-injection detectors, including Prompt Shields, under adaptive
        attacks. Treat this guardrail as a useful defense-in-depth signal, not a complete
        mitigation.

    For more information, see:

    - [Azure AI Content Safety: Prompt Shields (concept)](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/concepts/jailbreak-detection)
    - [Quickstart: use Prompt Shields](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/quickstart-jailbreak)
    - [Content Safety REST API reference (text jailbreak / Prompt Shields)](https://learn.microsoft.com/en-us/rest/api/contentsafety/text-operations/detect-text-jailbreak)
    - [Defending Against Indirect Prompt Injection Attacks With Spotlighting (arXiv:2403.14720)](https://arxiv.org/abs/2403.14720)
    - [Evaluating hosted prompt-injection detectors under adaptive attacks (arXiv:2504.11168)](https://arxiv.org/pdf/2504.11168)

    Args:
        endpoint: Azure Content Safety endpoint URL. Falls back to ``CONTENT_SAFETY_ENDPOINT``
            env var.
        api_key: Azure Content Safety API key. Falls back to ``CONTENT_SAFETY_KEY`` env var.

    """

    SUPPORTED_MODELS: ClassVar = ["azure-prompt-shields"]

    METADATA: ClassVar[GuardrailMetadata] = GUARDRAIL_METADATA[GuardrailName.AZURE_PROMPT_SHIELDS]

    def __init__(self, endpoint: str | None = None, api_key: str | None = None) -> None:
        """Initialize the Azure Prompt Shields guardrail.

        Args:
            endpoint: Azure Content Safety endpoint URL. If ``None``, the value is read from
                the ``CONTENT_SAFETY_ENDPOINT`` environment variable.
            api_key: Azure Content Safety API key. If ``None``, the value is read from the
                ``CONTENT_SAFETY_KEY`` environment variable.

        Raises:
            KeyError: If neither ``api_key`` nor ``CONTENT_SAFETY_KEY`` is set, or neither
                ``endpoint`` nor ``CONTENT_SAFETY_ENDPOINT`` is set.

        """
        if api_key:
            self._credential = AzureKeyCredential(api_key)
        else:
            try:
                self._credential = AzureKeyCredential(os.environ["CONTENT_SAFETY_KEY"])
            except KeyError as e:
                msg = (
                    "CONTENT_SAFETY_KEY environment variable is not set. "
                    "Either provide an api_key or set the environment variable."
                )
                raise KeyError(msg) from e

        if not endpoint:
            try:
                endpoint = os.environ["CONTENT_SAFETY_ENDPOINT"]
            except KeyError as e:
                msg = (
                    "CONTENT_SAFETY_ENDPOINT environment variable is not set. "
                    "Either provide an endpoint or set the environment variable."
                )
                raise KeyError(msg) from e

        self.endpoint = endpoint.rstrip("/")
        self._api_key = self._credential.key

    def validate(
        self,
        user_prompt: str | None = None,
        documents: list[str] | None = None,
    ) -> GuardrailOutput:
        """Detect direct and indirect prompt-injection attacks via Azure Prompt Shields.

        At least one of ``user_prompt`` or ``documents`` must be provided. The guardrail is
        considered invalid (``valid=False``) if Azure flags an attack in the user prompt or
        in **any** of the supplied documents.

        Args:
            user_prompt: End-user prompt to scan for direct prompt-injection / jailbreak
                attempts. If ``None``, only documents are analyzed.
            documents: Auxiliary documents (e.g. retrieved context, tool outputs) to scan
                for indirect (data-borne) prompt-injection. If ``None``, only the user
                prompt is analyzed.

        Returns:
            GuardrailOutput where ``valid`` is ``True`` iff no attack was detected anywhere,
            ``extra`` contains the per-field detection booleans, ``categories`` holds per-source
            verdicts (user prompt and each document), and ``score`` is ``1.0`` when any attack is
            detected and ``0.0`` otherwise.

        Raises:
            ValueError: If both ``user_prompt`` and ``documents`` are ``None``.
            RuntimeError: If the Azure REST API returns a non-success status code.

        """
        if user_prompt is None and documents is None:
            msg = "At least one of `user_prompt` or `documents` must be provided."
            raise ValueError(msg)

        start = time.perf_counter()
        response = self._call_shield_prompt(user_prompt=user_prompt, documents=documents)

        # Fail CLOSED on a malformed response (mirrors the #178 convention):
        # treat an unparsable Azure payload as unsafe rather than silently clean.
        # Only validate the surfaces actually requested.
        user_prompt_attack: bool | None = None
        if user_prompt is not None:
            user_prompt_analysis = response.get("userPromptAnalysis")
            if not isinstance(user_prompt_analysis, dict) or "attackDetected" not in user_prompt_analysis:
                return self._malformed_output(response, start)
            user_prompt_attack = bool(user_prompt_analysis.get("attackDetected", False))

        documents_attacks: list[bool] | None = None
        if documents is not None:
            documents_analysis = response.get("documentsAnalysis")
            if (
                not isinstance(documents_analysis, list)
                or len(documents_analysis) != len(documents)
                or not all(isinstance(item, dict) and "attackDetected" in item for item in documents_analysis)
            ):
                return self._malformed_output(response, start)
            documents_attacks = [bool(item.get("attackDetected", False)) for item in documents_analysis]

        any_doc_attack = bool(documents_attacks and any(documents_attacks))
        any_user_attack = bool(user_prompt_attack)
        attack_detected = any_user_attack or any_doc_attack

        categories: list[CategoryResult] = []
        if user_prompt_attack is not None:
            categories.append(CategoryResult(name="user_prompt", triggered=user_prompt_attack))
        for i, flag in enumerate(documents_attacks or []):
            categories.append(CategoryResult(name=f"document_{i}", triggered=flag))

        extra: dict[str, Any] = {
            "user_prompt_attack_detected": user_prompt_attack,
            "documents_attacks_detected": documents_attacks,
        }
        score = 1.0 if attack_detected else 0.0
        return GuardrailOutput(
            valid=not attack_detected,
            score=score,
            categories=categories,
            extra=extra,
            raw=response,
            usage=GuardrailUsage(latency_ms=(time.perf_counter() - start) * 1000.0),
        )

    def _malformed_output(self, response: dict[str, Any], start: float) -> GuardrailOutput:
        """Build a fail-closed ``GuardrailOutput`` for an unparsable Azure response.

        Mirrors the #178 fail-closed convention: an unparsable Azure payload is
        treated as unsafe (``valid=False``, ``score=1.0``) and flagged via
        ``extra["parse_failure"]`` rather than silently passing as clean.

        Args:
            response: The decoded (but malformed) Azure REST response.
            start: ``time.perf_counter()`` value captured before the request, for latency.

        Returns:
            GuardrailOutput with ``valid=False`` and ``extra["parse_failure"]=True``.

        """
        extra: dict[str, Any] = {
            "parse_failure": True,
            "user_prompt_attack_detected": None,
            "documents_attacks_detected": None,
        }
        return GuardrailOutput(
            valid=False,
            score=1.0,
            extra=extra,
            raw=response,
            usage=GuardrailUsage(latency_ms=(time.perf_counter() - start) * 1000.0),
        )

    def _call_shield_prompt(
        self,
        user_prompt: str | None,
        documents: list[str] | None,
    ) -> dict[str, Any]:
        """Issue a POST to the Prompt Shields REST endpoint and return the parsed JSON.

        Args:
            user_prompt: User prompt to include in the request, or ``None`` to omit.
            documents: Document list to include in the request, or ``None`` to omit.

        Returns:
            Decoded JSON response from the Azure REST API.

        Raises:
            RuntimeError: If the API returns a non-2xx status code.

        """
        url = f"{self.endpoint}/contentsafety/text:shieldPrompt?api-version={_API_VERSION}"
        headers = {
            "Ocp-Apim-Subscription-Key": self._api_key,
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {}
        if user_prompt is not None:
            payload["userPrompt"] = user_prompt
        if documents is not None:
            payload["documents"] = documents

        response = requests.post(url, headers=headers, json=payload)
        if response.status_code // 100 != 2:
            msg = f"Azure Prompt Shields request failed with status code {response.status_code}: {response.text}"
            raise RuntimeError(msg)
        parsed: dict[str, Any] = response.json()
        return parsed
