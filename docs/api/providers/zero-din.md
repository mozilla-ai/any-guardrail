# ZeroDinProvider

Execution provider for 0DIN's hosted SusFactor scoring API.

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

## Constructor

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `api_key` | `str | None` | No | `None` | 0DIN Portal API key; falls back to ``ODIN_API_KEY``. |
| `jwt` | `str | None` | No | `None` | Optional pre-minted 0DIN Defense JWT, used verbatim. |
| `base_url` | `str` | No | `"https://defense.0din.ai"` | Scoring host. Defaults to ``https://defense.0din.ai``. |
| `auth_endpoint` | `str` | No | `"https://0din.ai/api/v1/access_tokens"` | Token-minting URL. Defaults to ``https://0din.ai/api/v1/access_tokens``. |
| `request_timeout` | `float` | No | `30.0` | Per-request timeout in seconds. Defaults to 30. |

Configure the provider. Performs no network I/O — see ``load_model``.

## load_model

Resolve credentials and mint an initial JWT, so a bad key fails at construction.

**Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model_id` | `str` | Yes | — | The service identifier this provider serves; must be ``"susfactor-api"``. |

**Returns:** `None`

## pre_process

Wrap the prompt into the scoring request body.

0DIN tokenizes and chunks server-side, so the only client-side preparation is
shaping the JSON payload.

**Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `input_text` | `str` | Yes | — | The prompt to score. |

**Returns:** `GuardrailPreprocessOutput[AnyDict]`

## infer

POST the prompt to ``/api/v1/sus`` and return its suspicion score.

**Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model_inputs` | `GuardrailPreprocessOutput[AnyDict]` | Yes | — | The wrapped ``{"prompt": ...}`` payload from ``pre_process``. |

**Returns:** `GuardrailInferenceOutput[AnyDict]`

## close

Close the underlying HTTP session. Idempotent.

**Returns:** `None`
