import os
import time
from typing import ClassVar

import requests

from any_guardrail.base import Guardrail, GuardrailOutput
from any_guardrail.types import AnyDict, CategoryResult, ChatMessages


class Qualifire(Guardrail):
    """Wraps the Qualifire evaluation API for multi-check input/output guardrailing.

    This is the hosted, pay-per-use counterpart to the locally-run
    :class:`~any_guardrail.guardrails.sentinel.sentinel.Sentinel`
    (``qualifire/prompt-injection-sentinel``) classifier: the same maker's
    proprietary judges, exposing prompt-injection detection alongside
    hallucination, grounding, PII, content-moderation, custom policy assertions,
    and topic-scoping checks in one ``/evaluate`` request.

    Which checks run is configured on the guardrail; by default only
    prompt-injection detection is enabled (the direct analogue of ``Sentinel``).

    Auth is via an API key. Obtain one from https://qualifire.ai/ (free tier with
    a monthly token allowance) and set it via ``QUALIFIRE_API_KEY`` or pass it
    directly. Point at a self-hosted deployment with ``base_url=`` or
    ``QUALIFIRE_BASE_URL``.

    ``GuardrailOutput`` mapping:
        - ``valid = no per-check result was flagged``. (The API's top-level
          ``status`` is a lifecycle field — e.g. ``"completed"`` — not a verdict,
          so the per-check ``flagged`` booleans drive the decision.)
        - ``score`` is the highest *risk* among flagged checks. Qualifire scores
          are ``0-100`` where higher = safer, so they are converted to the
          canonical risk axis as ``1 - score / 100``; ``0.0`` when nothing was
          flagged.
        - ``categories`` flattens every per-check result (``name`` =
          ``"<type>/<check>"``, ``triggered`` = the check flagged, ``score`` =
          the per-check risk, ``description`` = its label).
        - ``explanation`` joins the reasons of the flagged checks.
        - ``extra`` carries the overall ``status`` and the raw Qualifire
          ``score`` (0-100); ``raw`` is the full response body.
        - Fails closed when the response lacks ``evaluationResults``.

    Research backing:
        - Ivry & Nahum, *Sentinel: SOTA model to protect against prompt injections*
          (https://arxiv.org/abs/2506.05446, 2025), the paper behind the
          ``qualifire/prompt-injection-sentinel`` model the hosted API's
          prompt-injection check is built on.
        - Docs: https://docs.qualifire.ai/

    Args:
        api_key (str | None): Qualifire API key. Falls back to ``QUALIFIRE_API_KEY``.
        base_url (str | None): API base URL. Falls back to ``QUALIFIRE_BASE_URL``,
            then ``https://api.qualifire.ai``.
        prompt_injections (bool): Run the prompt-injection check. Defaults to ``True``.
        pii_check (bool): Run the PII check. Defaults to ``False``.
        hallucinations_check (bool): Run the hallucination check. Defaults to ``False``.
        grounding_check (bool): Run the grounding (RAG) check. Defaults to ``False``.
        content_moderation_check (bool): Run content moderation. Defaults to ``False``.
        assertions (list[str] | None): Custom natural-language policies to enforce.
        allowed_topics (list[str] | None): Restrict conversation to these topics.

    """

    SUPPORTED_MODELS: ClassVar = ["qualifire"]

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        prompt_injections: bool = True,
        pii_check: bool = False,
        hallucinations_check: bool = False,
        grounding_check: bool = False,
        content_moderation_check: bool = False,
        assertions: list[str] | None = None,
        allowed_topics: list[str] | None = None,
    ) -> None:
        """Initialize the Qualifire guardrail.

        Does not perform any network I/O — the API is only contacted on
        ``validate()``.
        """
        if api_key:
            self.api_key = api_key
        elif os.getenv("QUALIFIRE_API_KEY"):
            self.api_key = os.getenv("QUALIFIRE_API_KEY")  # type: ignore[assignment]
        else:
            msg = (
                "API key must be provided either as the `api_key=` parameter or through the "
                "QUALIFIRE_API_KEY environment variable. Sign up at https://qualifire.ai/ to obtain a key."
            )
            raise ValueError(msg)

        self.base_url = (base_url or os.getenv("QUALIFIRE_BASE_URL") or "https://api.qualifire.ai").rstrip("/")
        self.prompt_injections = prompt_injections
        self.pii_check = pii_check
        self.hallucinations_check = hallucinations_check
        self.grounding_check = grounding_check
        self.content_moderation_check = content_moderation_check
        self.assertions = assertions
        self.allowed_topics = allowed_topics

    def validate(
        self,
        input_text: str | None = None,
        output: str | None = None,
        messages: ChatMessages | None = None,
    ) -> GuardrailOutput:
        """Evaluate an input, output, and/or conversation with the configured checks.

        At least one of ``input_text``, ``output``, or ``messages`` must be given.

        Args:
            input_text (str | None): The model input to evaluate.
            output (str | None): The model output to evaluate.
            messages (ChatMessages | None): A chat-message list to evaluate.

        Returns:
            ``GuardrailOutput`` summarizing the checks (see the class docstring
            for the field mapping).

        """
        if input_text is None and output is None and not messages:
            msg = "At least one of `input_text`, `output`, or `messages` must be provided."
            raise ValueError(msg)

        start = time.perf_counter()
        params = self._pre_processing(input_text, output, messages)
        response = self._inference(params)
        result = self._post_processing(response)
        self._stamp_usage(result, (time.perf_counter() - start) * 1000.0)
        return result

    def _pre_processing(
        self,
        input_text: str | None,
        output: str | None,
        messages: ChatMessages | None,
    ) -> AnyDict:
        body: AnyDict = {
            "prompt_injections": self.prompt_injections,
            "pii_check": self.pii_check,
            "hallucinations_check": self.hallucinations_check,
            "grounding_check": self.grounding_check,
            "content_moderation_check": self.content_moderation_check,
        }
        if input_text is not None:
            body["input"] = input_text
        if output is not None:
            body["output"] = output
        if messages:
            body["messages"] = messages
        if self.assertions:
            body["assertions"] = self.assertions
        if self.allowed_topics:
            body["allowed_topics"] = self.allowed_topics
        return body

    def _inference(self, params: AnyDict) -> requests.Response:
        response = requests.post(
            f"{self.base_url}/api/v1/evaluation/evaluate",
            headers={"X-Qualifire-API-Key": self.api_key},
            json=params,
        )
        if response.status_code != 200:
            msg = f"Request to Qualifire API failed with status code {response.status_code}: {response.text}"
            raise ValueError(msg)
        return response

    @staticmethod
    def _to_risk(score: object) -> float | None:
        """Convert a Qualifire 0-100 (higher = safer) score to canonical risk in [0, 1]."""
        if not isinstance(score, (int, float)):
            return None
        return min(1.0, max(0.0, 1.0 - float(score) / 100.0))

    def _post_processing(self, response: requests.Response) -> GuardrailOutput:
        body = response.json()
        eval_results = body.get("evaluationResults") if isinstance(body, dict) else None
        if not isinstance(eval_results, list) or not eval_results:
            # Missing, null, non-list, or empty -> no parseable verdict; fail closed.
            return GuardrailOutput(valid=False, extra={"parse_failure": True}, raw=body)

        categories: list[CategoryResult] = []
        explanations: list[str] = []
        flagged_risks: list[float] = []
        any_flagged = False
        parsed_checks = 0

        for item in eval_results:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type") or "check"
            for check in item.get("results") or []:
                if not isinstance(check, dict):
                    continue
                parsed_checks += 1
                flagged = bool(check.get("flagged"))
                any_flagged = any_flagged or flagged
                risk = self._to_risk(check.get("score"))
                name = check.get("name") or item_type
                categories.append(
                    CategoryResult(
                        name=f"{item_type}/{name}",
                        description=check.get("label"),
                        triggered=flagged,
                        score=risk,
                    )
                )
                if flagged:
                    if risk is not None:
                        flagged_risks.append(risk)
                    reason = check.get("reason")
                    if isinstance(reason, str) and reason:
                        explanations.append(reason)

        if parsed_checks == 0:
            # Entries existed but none carried a parseable check -> no verdict; fail closed.
            return GuardrailOutput(valid=False, extra={"parse_failure": True}, raw=body)

        # The verdict is the per-check ``flagged`` booleans; ``status`` is a
        # lifecycle field (e.g. "completed"), not a pass/fail signal.
        return GuardrailOutput(
            valid=not any_flagged,
            score=max(flagged_risks) if flagged_risks else 0.0,
            explanation="\n\n".join(explanations) if explanations else None,
            categories=categories,
            extra={"status": body.get("status"), "qualifire_score": body.get("score")},
            raw=body,
        )
