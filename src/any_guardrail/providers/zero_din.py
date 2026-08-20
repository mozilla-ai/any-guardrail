"""Provider backed by 0DIN's hosted SusFactor scoring API.

0DIN Defense (https://0din.ai/docs/defense/api) serves the same SusFactor
prompt-injection classifier that ships as a gated ONNX model on HuggingFace,
but as a hosted REST service. This provider mints a short-lived JWT from a 0DIN
Portal API key and proxies scoring calls to it, so the ``Susfactor`` guardrail
can run without downloading — or being entitled to — the gated model repository.

Only ``requests`` is required, which is a core dependency: the hosted path works
on a bare ``pip install any-guardrail`` with no extras and no ``numpy``/``torch``.

Provider contract for ``Susfactor``: ``infer()`` returns
``{"chunk_scores": list[float], "raw": Any}``. The hosted API chunks long prompts
server-side and returns the maximum ``P(suspicious)`` across those chunks, so a
single-element ``chunk_scores`` is a faithful (not lossy) equivalent of the local
per-chunk list — ``max(S) >= t`` and ``any(s >= t for s in S)`` are the same
predicate, and only those two quantities reach ``GuardrailOutput``.
"""

from __future__ import annotations

import os
import threading
import time
from typing import TYPE_CHECKING, Any, ClassVar

import requests

from any_guardrail.providers.base import Provider
from any_guardrail.types import (
    AnyDict,
    GuardrailInferenceOutput,
    GuardrailPreprocessOutput,
)

if TYPE_CHECKING:
    from typing import Self

_DEFAULT_AUTH_URL = "https://0din.ai/api/v1/access_tokens"
_DEFAULT_BASE_URL = "https://defense.0din.ai"
_SCORE_PATH = "/api/v1/sus"
_API_KEY_ENV_VAR = "ODIN_API_KEY"

# The Portal returns ``expires_in: 900``; fall back to that when the field is absent
# or unusable, and renew this many seconds early to absorb clock skew and in-flight
# requests.
_DEFAULT_TTL_SECONDS = 900.0
_JWT_LEEWAY_SECONDS = 60.0

_HTTP_OK = 200
_HTTP_UNAUTHORIZED = 401

SUSFACTOR_API_MODEL_ID = "susfactor-api"
"""Service identifier reported as ``usage.model_id`` on the hosted path.

Mirrors ``Susfactor.SUPPORTED_MODELS[1]``; kept as a literal here so the provider
layer never imports from ``any_guardrail.guardrails`` (a unit test pins them equal).
"""


class ZeroDinProvider(Provider[AnyDict, AnyDict]):
    """Execution provider for 0DIN's hosted SusFactor scoring API.

    Pass it as ``AnyGuardrail.create(GuardrailName.SUSFACTOR, provider=ZeroDinProvider())``
    to run the ``Susfactor`` guardrail against 0DIN Defense instead of the gated local
    ONNX model — the guardrail class, its ``threshold``, and its ``GuardrailOutput``
    shape are unchanged.

    Authentication is a two-step exchange. A long-lived 0DIN Portal API key is posted
    to ``https://0din.ai/api/v1/access_tokens`` to mint a JWT valid for 900 seconds,
    and that JWT is sent as a bearer token to ``https://defense.0din.ai/api/v1/sus``.
    The provider caches the JWT, renews it a minute before expiry, and re-mints once
    if the service rejects it anyway (clock skew, revocation, key rotation).

    Unlike the REST guardrails in this codebase, requests carry an explicit
    ``request_timeout``: this provider sits in the request path, where hanging forever
    on a half-open connection is the worst available failure mode.

    Behavioral note: 0DIN chunks long prompts server-side with unpublished size/stride
    parameters and returns only the maximum score, whereas the local ONNX path uses a
    510-token window with 50-token overlap and exposes every chunk score. The two
    backends can therefore disagree on very long inputs.

    Args:
        api_key: 0DIN Portal API key. Falls back to the ``ODIN_API_KEY`` environment
            variable. Sign up for the SusFactor early-access beta at
            https://0din.ai/susfactor-trial to obtain one.
        jwt: A pre-minted 0DIN Defense JWT, for callers whose tokens come from a
            sidecar or vault. Used verbatim and never proactively renewed; supply
            ``api_key`` as well if you want automatic renewal.
        base_url: Scoring host. Defaults to ``https://defense.0din.ai``.
        auth_endpoint: Token-minting URL. Defaults to
            ``https://0din.ai/api/v1/access_tokens`` (a different host from the
            scoring one).
        request_timeout: Per-request timeout in seconds, applied to both the mint and
            the scoring call. Defaults to 30.

    """

    default_model_id: ClassVar[str] = SUSFACTOR_API_MODEL_ID
    """Which ``Susfactor.SUPPORTED_MODELS`` entry this provider serves."""

    def __init__(
        self,
        api_key: str | None = None,
        jwt: str | None = None,
        base_url: str = _DEFAULT_BASE_URL,
        auth_endpoint: str = _DEFAULT_AUTH_URL,
        request_timeout: float = 30.0,
    ) -> None:
        """Configure the provider. Performs no network I/O — see ``load_model``.

        Args:
            api_key: 0DIN Portal API key; falls back to ``ODIN_API_KEY``.
            jwt: Optional pre-minted 0DIN Defense JWT, used verbatim.
            base_url: Scoring host. Defaults to ``https://defense.0din.ai``.
            auth_endpoint: Token-minting URL. Defaults to
                ``https://0din.ai/api/v1/access_tokens``.
            request_timeout: Per-request timeout in seconds. Defaults to 30.

        """
        self.base_url = base_url.rstrip("/")
        self.auth_endpoint = auth_endpoint
        self.request_timeout = request_timeout
        self.model_id: str | None = None

        self._api_key = api_key or os.getenv(_API_KEY_ENV_VAR)
        self._jwt = jwt
        # An injected JWT has no known expiry, so never renew it proactively; a
        # 0.0 deadline means "cold cache, mint on first use".
        self._jwt_expiry = float("inf") if jwt else 0.0
        self._lock = threading.Lock()
        self._session = requests.Session()
        self._closed = False

    def load_model(self, model_id: str, **kwargs: Any) -> None:
        """Resolve credentials and mint an initial JWT, so a bad key fails at construction.

        Args:
            model_id: The service identifier this provider serves; must be
                ``"susfactor-api"``.
            **kwargs: Ignored — the hosted API exposes no load-time knobs.

        Raises:
            ValueError: If ``model_id`` names a different backend, if neither
                ``api_key`` nor ``jwt`` resolves, or if the mint request fails.

        """
        del kwargs
        if model_id != self.default_model_id:
            msg = (
                f"{type(self).__name__} serves {self.default_model_id!r}, not {model_id!r}. "
                f"Omit model_id (it is resolved from the provider) or pass "
                f"model_id={self.default_model_id!r}."
            )
            raise ValueError(msg)
        if self._api_key is None and self._jwt is None:
            msg = (
                "A 0DIN Portal API key must be provided either as the `api_key=` parameter or "
                f"through the {_API_KEY_ENV_VAR} environment variable (or supply a pre-minted "
                "`jwt=`). Sign up for the SusFactor early-access beta at "
                "https://0din.ai/susfactor-trial to obtain a key."
            )
            raise ValueError(msg)
        self.model_id = model_id
        if self._jwt is None:
            self._bearer()

    def pre_process(self, input_text: str, **kwargs: Any) -> GuardrailPreprocessOutput[AnyDict]:
        """Wrap the prompt into the scoring request body.

        0DIN tokenizes and chunks server-side, so the only client-side preparation is
        shaping the JSON payload.

        Args:
            input_text: The prompt to score.
            **kwargs: Ignored — the hosted API exposes no per-request knobs.

        Returns:
            GuardrailPreprocessOutput wrapping ``{"prompt": input_text}``.

        """
        del kwargs
        return GuardrailPreprocessOutput(data={"prompt": input_text})

    def infer(self, model_inputs: GuardrailPreprocessOutput[AnyDict]) -> GuardrailInferenceOutput[AnyDict]:
        """POST the prompt to ``/api/v1/sus`` and return its suspicion score.

        Args:
            model_inputs: The wrapped ``{"prompt": ...}`` payload from ``pre_process``.

        Returns:
            GuardrailInferenceOutput wrapping ``{"chunk_scores": [score], "raw": <response JSON>}``.
            ``chunk_scores`` is empty when the response carries no usable numeric
            ``score``, which the guardrail turns into a fail-closed verdict.

        Raises:
            RuntimeError: If ``load_model()`` has not been called.
            ValueError: If the API responds with a non-200 status.

        """
        if self.model_id is None:
            msg = "load_model() must be called before infer()"
            raise RuntimeError(msg)

        prompt = model_inputs.data["prompt"]
        response = self._score(prompt, force_mint=False)
        if response.status_code == _HTTP_UNAUTHORIZED and self._api_key is not None:
            # The cached JWT was rejected even though we thought it was live. Scoring
            # has no side effects, so re-mint and retry exactly once — never a loop.
            response = self._score(prompt, force_mint=True)
        if response.status_code != _HTTP_OK:
            msg = f"Request to the 0DIN SusFactor API failed with status code {response.status_code}: {response.text}"
            raise ValueError(msg)

        body = response.json()
        score = body.get("score") if isinstance(body, dict) else None
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            # Present but unparsable: hand the guardrail an empty score list so it
            # fails closed rather than inventing a verdict.
            return GuardrailInferenceOutput(data={"chunk_scores": [], "raw": body})
        return GuardrailInferenceOutput(data={"chunk_scores": [float(score)], "raw": body})

    def close(self) -> None:
        """Close the underlying HTTP session. Idempotent."""
        if self._closed:
            return
        self._session.close()
        self._closed = True

    def __enter__(self) -> Self:
        """Enter the context manager, returning this provider."""
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Close the HTTP session on context exit."""
        self.close()

    def _score(self, prompt: str, *, force_mint: bool) -> requests.Response:
        """POST one prompt to the scoring endpoint with a live bearer token."""
        return self._session.post(
            f"{self.base_url}{_SCORE_PATH}",
            headers={
                "Authorization": f"Bearer {self._bearer(force=force_mint)}",
                "Content-Type": "application/json",
            },
            json={"prompt": prompt},
            timeout=self.request_timeout,
        )

    def _bearer(self, *, force: bool = False) -> str:
        """Return a live JWT, minting one when the cache is cold, expired, or forced.

        The lock is held across the mint call on purpose: it collapses a thundering
        herd of concurrent ``validate()`` calls into a single mint (roughly one per
        14 minutes). The scoring request itself is never made under the lock.
        """
        with self._lock:
            if not force and self._jwt is not None and time.monotonic() < self._jwt_expiry:
                return self._jwt
            if self._api_key is None:
                msg = (
                    "The supplied jwt= was rejected or has expired, and no API key is available "
                    f"to renew it. Pass api_key= (or set {_API_KEY_ENV_VAR}) for automatic renewal."
                )
                raise ValueError(msg)
            return self._mint_locked(self._api_key)

    def _mint_locked(self, api_key: str) -> str:
        """Exchange the Portal API key for a JWT. Caller must hold ``self._lock``."""
        response = self._session.post(
            self.auth_endpoint,
            headers={"Authorization": api_key},
            timeout=self.request_timeout,
        )
        if response.status_code != _HTTP_OK:
            msg = (
                f"Request to the 0DIN access-token endpoint failed with status code "
                f"{response.status_code}: {response.text}"
            )
            raise ValueError(msg)

        body = response.json()
        minted = body.get("token") if isinstance(body, dict) else None
        if not isinstance(minted, str) or not minted:
            msg = f"The 0DIN access-token endpoint returned no usable 'token' field: {body!r}"
            raise ValueError(msg)

        ttl = body.get("expires_in")
        seconds = float(ttl) if isinstance(ttl, (int, float)) and not isinstance(ttl, bool) else _DEFAULT_TTL_SECONDS
        self._jwt = minted
        # monotonic(), not time(): an NTP step or a resume-from-sleep must never make
        # a dead token look live.
        self._jwt_expiry = time.monotonic() + max(seconds - _JWT_LEEWAY_SECONDS, 1.0)
        return minted
