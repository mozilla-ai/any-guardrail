import os
import time
from typing import TYPE_CHECKING, Any, ClassVar

from any_guardrail.base import Guardrail, GuardrailOutput
from any_guardrail.types import AnyDict, CategoryResult, SpanResult

if TYPE_CHECKING:
    from ibm_watsonx_ai import APIClient

_IMPORT_ERROR_HINT = (
    "WatsonxGuardian requires the `ibm-watsonx-ai` package. Install it with "
    "`pip install any-guardrail[watsonx]` (or `pip install ibm-watsonx-ai`)."
)


class WatsonxGuardian(Guardrail):
    """Wraps IBM watsonx.ai's Text Detection / ``Guardian`` moderation API.

    This is the hosted, pay-per-use counterpart to the locally-run
    :class:`~any_guardrail.guardrails.granite_guardian.granite_guardian.GraniteGuardian`
    guardrail: the same Granite Guardian risk-detection family, served as a
    purpose-built detection endpoint instead of running the weights yourself.

    The ``Guardian`` class (from the ``ibm-watsonx-ai`` SDK) screens text against
    a configurable set of detectors. The default ``granite_guardian`` detector
    covers the Granite Guardian risk catalogue (harm, social bias, violence,
    jailbreak, profanity, sexual content, plus RAG groundedness / relevance);
    ``hap`` (hate-abuse-profanity) and ``pii`` detectors are also available. Each
    detector returns zero or more *detections*, each locating a risky span with a
    score.

    Auth is via an IBM Cloud IAM API key plus a region URL and a project (or
    space). Obtain a key and project from https://dataplatform.cloud.ibm.com/ and
    set them via ``WATSONX_APIKEY`` / ``WATSONX_URL`` / ``WATSONX_PROJECT_ID``
    (or ``WATSONX_SPACE_ID``), or pass them directly. A free Lite plan is
    available.

    ``GuardrailOutput`` mapping:
        - ``valid = no detections were returned`` (the detection API only returns
          detections at or above the configured threshold).
        - ``score`` is the highest detection score; ``0.0`` when nothing was
          detected.
        - ``categories`` lists one ``CategoryResult`` per detection (``name`` =
          the detected risk, ``triggered=True``, ``score`` = the detection score).
        - ``spans`` lists one ``SpanResult`` per detection that carries character
          offsets (watsonx detections locate the flagged substring).
        - ``raw`` is the full response dict from ``Guardian.detect``.

    Research backing:
        - Padhi et al., *Granite Guardian* (https://arxiv.org/abs/2412.07724, 2024).
        - IBM tutorial: https://www.ibm.com/think/tutorials/llm-safeguards-granite-guardian-risk-detection
        - SDK reference: https://ibm.github.io/watsonx-ai-python-sdk/fm_text_detection.html

    Args:
        api_key (str | None): IBM Cloud IAM API key. Falls back to ``WATSONX_APIKEY``.
        url (str | None): watsonx.ai region endpoint (e.g.
            ``https://us-south.ml.cloud.ibm.com``). Falls back to ``WATSONX_URL``.
        project_id (str | None): watsonx project ID. Falls back to
            ``WATSONX_PROJECT_ID``. One of ``project_id`` / ``space_id`` is required.
        space_id (str | None): watsonx deployment space ID. Falls back to
            ``WATSONX_SPACE_ID``.
        detectors (dict | None): Detector configuration forwarded to ``Guardian``.
            Defaults to ``{"granite_guardian": {}}``. Pass e.g.
            ``{"granite_guardian": {"threshold": 0.6}, "pii": {}}`` to tune it.
        api_client (APIClient | None): A pre-built ``ibm_watsonx_ai.APIClient``.
            When supplied, the credential arguments above are ignored and the
            client is used as-is (useful for shared clients or testing).

    """

    SUPPORTED_MODELS: ClassVar = ["granite_guardian"]

    def __init__(
        self,
        api_key: str | None = None,
        url: str | None = None,
        project_id: str | None = None,
        space_id: str | None = None,
        detectors: AnyDict | None = None,
        api_client: "APIClient | None" = None,
    ) -> None:
        """Initialize the guardrail and build the watsonx ``Guardian`` client.

        Building the client performs IAM authentication, so unlike the pure-REST
        API guardrails this constructor does contact IBM Cloud (unless a
        pre-built ``api_client`` is supplied).
        """
        self.model_id = "granite_guardian"
        self.detectors = detectors if detectors is not None else {"granite_guardian": {}}

        # Validate credentials before importing the SDK so a user who forgot a
        # key gets a clear ValueError rather than an ImportError.
        if api_client is None:
            self.api_key = api_key or os.getenv("WATSONX_APIKEY")
            self.url = url or os.getenv("WATSONX_URL")
            self.project_id = project_id or os.getenv("WATSONX_PROJECT_ID")
            self.space_id = space_id or os.getenv("WATSONX_SPACE_ID")
            if not self.api_key:
                msg = (
                    "API key must be provided either as the `api_key=` parameter or through the "
                    "WATSONX_APIKEY environment variable."
                )
                raise ValueError(msg)
            if not self.url:
                msg = (
                    "watsonx URL must be provided either as the `url=` parameter or through the "
                    "WATSONX_URL environment variable (e.g. https://us-south.ml.cloud.ibm.com)."
                )
                raise ValueError(msg)
            if not self.project_id and not self.space_id:
                msg = (
                    "A project or space is required: provide `project_id=`/`space_id=` or set "
                    "WATSONX_PROJECT_ID / WATSONX_SPACE_ID."
                )
                raise ValueError(msg)

        try:
            from ibm_watsonx_ai import APIClient, Credentials
            from ibm_watsonx_ai.foundation_models.moderations import Guardian
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(_IMPORT_ERROR_HINT) from exc

        if api_client is None:
            credentials = Credentials(url=self.url, api_key=self.api_key)
            api_client = APIClient(credentials, project_id=self.project_id, space_id=self.space_id)

        self.guardian = Guardian(api_client=api_client, detectors=self.detectors)

    def validate(self, content: str) -> GuardrailOutput:
        """Screen ``content`` against the configured watsonx detectors.

        Args:
            content (str): The text to screen.

        Returns:
            ``GuardrailOutput`` with ``valid = no detections``, ``score`` set to
            the highest detection score, ``categories`` / ``spans`` describing
            each detection, and ``raw`` holding the full response. When the
            response cannot be parsed, the output fails closed (``valid=False``
            with ``extra={"parse_failure": True}``).

        """
        start = time.perf_counter()
        response = self.guardian.detect(text=content)
        result = self._post_processing(response)
        self._stamp_usage(result, (time.perf_counter() - start) * 1000.0)
        return result

    def _post_processing(self, response: Any) -> GuardrailOutput:
        detections = response.get("detections") if isinstance(response, dict) else None
        if not isinstance(detections, list):
            # ``detections`` absent, null, or a non-list -> present but cannot be parsed; fail closed.
            return GuardrailOutput(valid=False, extra={"parse_failure": True}, raw=response)

        scores = [float(d["score"]) for d in detections if isinstance(d, dict) and d.get("score") is not None]

        categories: list[CategoryResult] = []
        spans: list[SpanResult] = []
        for detection in detections:
            if not isinstance(detection, dict):
                continue
            name = detection.get("detection") or detection.get("detection_type") or "unknown"
            score = float(detection["score"]) if detection.get("score") is not None else None
            categories.append(
                CategoryResult(
                    name=name,
                    description=detection.get("detection_type"),
                    triggered=True,
                    score=score,
                )
            )
            if detection.get("start") is not None and detection.get("end") is not None:
                spans.append(
                    SpanResult(
                        start=int(detection["start"]),
                        end=int(detection["end"]),
                        label=name,
                        score=score,
                    )
                )

        if scores:
            overall_score: float | None = max(scores)
        elif detections:
            overall_score = None  # flagged but unscored: don't claim 0.0 (= no) risk
        else:
            overall_score = 0.0

        return GuardrailOutput(
            valid=not detections,
            score=overall_score,
            categories=categories,
            spans=spans or None,
            raw=response,
        )
