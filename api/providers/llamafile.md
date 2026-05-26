# LlamafileProvider

Run inference through a local ``llamafile`` binary's HTTP server.

The provider spawns the binary as a subprocess listening on ``--host``/
``--port`` (server mode is implicit when a port is given in llamafile
0.10+), with ``--no-webui`` to suppress the UI, then polls ``GET /health``
for readiness and issues ``POST /v1/chat/completions`` calls. Output is
normalized to the same shape :meth:`HuggingFaceProvider.generate_chat`
returns so guardrails are provider-agnostic.

The provider implements the context manager protocol for deterministic
cleanup of the spawned subprocess::

    with LlamafileProvider() as provider:
        guardrail = GraniteGuardian(
            criteria=GraniteGuardianRisk.HARM, provider=provider
        )
        result = guardrail.validate("hello")
    # subprocess is terminated here, even if validate() raised.

Outside a ``with`` block the provider still cleans up via ``atexit`` on
interpreter exit, so notebook and REPL usage works without explicit
teardown. Call ``provider.close()`` directly to release the port early.

Args:
    binary_path: Path to a pre-downloaded ``.llamafile``. If omitted, the
        artifact is auto-downloaded — first by trying ``repo_id``/``filename``
        if both were supplied, otherwise by looking up the ``model_id``
        passed to ``load_model`` in the curated
        :data:`~any_guardrail.providers._llamafile_artifacts.LLAMAFILE_ARTIFACTS`
        map.
    repo_id: Power-user override for the HuggingFace repo containing the
        llamafile. Used together with ``filename``.
    filename: Power-user override for the artifact filename inside
        ``repo_id``. Used together with ``repo_id``.
    port: TCP port to bind the llamafile HTTP server. Defaults to a
        kernel-chosen free port.
    host: Bind address. Defaults to ``"127.0.0.1"``.
    startup_timeout: Seconds to wait for the server to become ready.
        Llamafiles can take ~30s to memory-map and warm up; the default
        is generous.
    request_timeout: Per-request timeout for ``/v1/chat/completions``.
    cache_dir: Directory passed to ``hf_hub_download`` for auto-downloaded
        binaries.
    n_gpu_layers: Optional number of model layers to offload to GPU. Passed
        as ``--n-gpu-layers``. ``None`` (default) lets llamafile decide.
    context_size: Optional context window size. Passed as ``--ctx-size``.
    extra_args: Optional list of additional command-line arguments appended
        after the standard server flags. Use this for advanced llamafile
        flags not surfaced above.

## Constructor

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `binary_path` | `str | None` | No | `None` |
| `repo_id` | `str | None` | No | `None` |
| `filename` | `str | None` | No | `None` |
| `port` | `int | None` | No | `None` |
| `host` | `str` | No | `"127.0.0.1"` |
| `startup_timeout` | `float` | No | `120.0` |
| `request_timeout` | `float` | No | `120.0` |
| `cache_dir` | `str | None` | No | `None` |
| `n_gpu_layers` | `int | None` | No | `None` |
| `context_size` | `int | None` | No | `None` |
| `extra_args` | `list[str] | None` | No | `None` |

Initialize the llamafile provider.

## load_model

Resolve the llamafile binary for ``model_id`` and start its HTTP server.

If we auto-pick the port and the subprocess fails to come up (e.g.
another process grabbed the port between our `_free_port()` probe
and the binary's `bind()`), retry up to :attr:`_BIND_RACE_RETRIES`
times with a fresh port. When the caller pinned a port via the
``port=`` constructor argument, no retry: surface the failure
immediately.

**Parameters**

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `model_id` | `str` | Yes | — |

**Returns:** `None`

## pre_process

Not supported — llamafile is a chat-style backend.

Use :meth:`generate_chat` instead. Decoder-LLM guardrails like
:class:`GraniteGuardian` route through ``generate_chat`` automatically.

**Returns:** `GuardrailPreprocessOutput[AnyDict]`

## infer

Not supported — llamafile is a chat-style backend.

Use :meth:`generate_chat` instead.

**Parameters**

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `model_inputs` | `GuardrailPreprocessOutput[AnyDict]` | Yes | — |

**Returns:** `GuardrailInferenceOutput[AnyDict]`

## close

Terminate the llamafile subprocess. Idempotent.

**Returns:** `None`
