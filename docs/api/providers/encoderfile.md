# EncoderfileProvider

Run inference through a local ``encoderfile`` binary's HTTP server.

The provider spawns the binary as a subprocess, polls for readiness, then
issues ``POST /predict`` calls. Output is normalized to the same shape
HuggingFaceProvider returns so downstream guardrails are provider-agnostic.

The provider implements the context manager protocol for deterministic
cleanup of the spawned subprocess::

    with EncoderfileProvider() as provider:
        guardrail = Protectai(provider=provider)
        result = guardrail.validate("hello")
    # subprocess is terminated here, even if validate() raised.

Outside a ``with`` block the provider still cleans up via ``atexit`` on
interpreter exit, so notebook and REPL usage works without explicit
teardown. Call ``provider.close()`` directly to release the port early.

Args:
    binary_path: Path to a pre-built ``.encoderfile``. If omitted, the
        platform-appropriate artifact is auto-downloaded from
        ``mozilla-ai/encoderfile`` using the model_id passed to
        ``load_model``. Mutually exclusive with ``base_url``.
    base_url: External-server mode. Point at an encoderfile server you spun
        up yourself (e.g. ``"http://localhost:9999"``). When set, the
        provider skips download + subprocess spawn entirely; ``load_model``
        only polls the server for readiness, and ``close()`` is a no-op.
        Mutually exclusive with ``binary_path``, ``port``, and a
        non-default ``encoderfile_repo``. Must start with ``http://`` or
        ``https://``.
    port: TCP port to bind the encoderfile HTTP server. Defaults to a
        kernel-chosen free port. Mutually exclusive with ``base_url``.
    host: Bind address. Defaults to ``"127.0.0.1"``.
    startup_timeout: Seconds to wait for the server to become ready. Also
        applies to external-server readiness polling.
    request_timeout: Per-request timeout for ``/predict`` calls.
    cache_dir: Directory passed to ``hf_hub_download`` for auto-downloaded
        binaries.
    encoderfile_repo: Override the source HF repo. Defaults to
        ``mozilla-ai/encoderfile``. Mutually exclusive with ``base_url``
        when set to a non-default value.

## Constructor

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `binary_path` | `str | None` | No | `None` |
| `base_url` | `str | None` | No | `None` |
| `port` | `int | None` | No | `None` |
| `host` | `str` | No | `"127.0.0.1"` |
| `startup_timeout` | `float` | No | `60.0` |
| `request_timeout` | `float` | No | `60.0` |
| `cache_dir` | `str | None` | No | `None` |
| `encoderfile_repo` | `str` | No | `"mozilla-ai/encoderfile"` |

Initialize the encoderfile provider.

## load_model

Load the encoderfile binary for ``model_id`` and start its HTTP server.

If we auto-pick the port and the subprocess fails to come up (e.g.
another process grabbed the port between our `_free_port()` probe and
the binary's `bind()`), retry up to :attr:`_BIND_RACE_RETRIES` times
with a fresh port. When the caller pinned a port via the
``port=`` constructor argument, no retry: surface the failure
immediately.

In external-server mode (``base_url`` supplied to the constructor),
the binary lookup and subprocess spawn are skipped — the provider
only polls the user's server for readiness.

**Parameters**

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `model_id` | `str` | Yes | — |

**Returns:** `None`

## pre_process

Wrap raw text into the encoderfile request body.

Encoderfile does its own tokenization inside the binary; the only
client-side preparation is shaping the JSON payload.

**Parameters**

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `input_text` | `str | list[str]` | Yes | — |

**Returns:** `GuardrailPreprocessOutput[AnyDict]`

## infer

POST the preprocessed payload to the running encoderfile server.

Returns the same uniform shape as HuggingFaceProvider: ``logits``,
``scores``, ``predicted_indices``, ``predicted_labels``, ``labels``.
``labels`` is ``None`` because the encoderfile ``/predict`` response
only carries the predicted label, not the full ordered label list.

**Parameters**

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `model_inputs` | `GuardrailPreprocessOutput[AnyDict]` | Yes | — |

**Returns:** `GuardrailInferenceOutput[AnyDict]`

## close

Terminate the encoderfile subprocess. Idempotent.

In external-server mode there is no subprocess to terminate and
``self.base_url`` is preserved so the provider stays reusable.

**Returns:** `None`
