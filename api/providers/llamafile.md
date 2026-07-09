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

## Constructor

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `binary_path` | `str | None` | No | `None` | Path to a pre-downloaded ``.llamafile``. If omitted, the artifact is auto-downloaded — first by trying ``repo_id``/``filename`` if both were supplied, otherwise by looking up the ``model_id`` passed to ``load_model`` in the curated :data:`~any_guardrail.providers._llamafile_artifacts.LLAMAFILE_ARTIFACTS` map. Mutually exclusive with ``base_url``. |
| `repo_id` | `str | None` | No | `None` | Power-user override for the HuggingFace repo containing the llamafile. Used together with ``filename``. Mutually exclusive with ``base_url``. |
| `filename` | `str | None` | No | `None` | Power-user override for the artifact filename inside ``repo_id``. Used together with ``repo_id``. Mutually exclusive with ``base_url``. |
| `base_url` | `str | None` | No | `None` | External-server mode. Point at a llamafile server you spun up yourself (e.g. ``"http://localhost:9999"``). When set, the provider skips download + subprocess spawn entirely; ``load_model`` only polls the server for readiness, and ``close()`` is a no-op. Mutually exclusive with ``binary_path``, ``repo_id``/``filename``, ``port``, ``n_gpu_layers``, ``context_size``, and ``extra_args``. Must start with ``http://`` or ``https://``. |
| `port` | `int | None` | No | `None` | TCP port to bind the llamafile HTTP server. Defaults to a kernel-chosen free port. Mutually exclusive with ``base_url``. |
| `host` | `str` | No | `"127.0.0.1"` | Bind address. Defaults to ``"127.0.0.1"``. |
| `startup_timeout` | `float` | No | `120.0` | Seconds to wait for the server to become ready. Llamafiles can take ~30s to memory-map and warm up; the default is generous. Also applies to external-server readiness polling. |
| `request_timeout` | `float` | No | `120.0` | Per-request timeout for ``/v1/chat/completions``. |
| `cache_dir` | `str | None` | No | `None` | Directory passed to ``hf_hub_download`` for auto-downloaded binaries. |
| `n_gpu_layers` | `int | None` | No | `None` | Optional number of model layers to offload to GPU. Passed as ``--n-gpu-layers``. ``None`` (default) lets llamafile decide. Mutually exclusive with ``base_url``. |
| `context_size` | `int | None` | No | `None` | Optional context window size. Passed as ``--ctx-size``. Mutually exclusive with ``base_url``. |
| `extra_args` | `list[str] | None` | No | `None` | Optional list of additional command-line arguments appended after the standard server flags. Use this for advanced llamafile flags not surfaced above. Mutually exclusive with ``base_url``. |

Initialize the llamafile provider.

## load_model

Resolve the llamafile binary for ``model_id`` and start its HTTP server.

If we auto-pick the port and the subprocess fails to come up (e.g.
another process grabbed the port between our `_free_port()` probe
and the binary's `bind()`), retry up to :attr:`_BIND_RACE_RETRIES`
times with a fresh port. When the caller pinned a port via the
``port=`` constructor argument, no retry: surface the failure
immediately.

In external-server mode (``base_url`` supplied to the constructor),
the binary lookup and subprocess spawn are skipped — the provider
only polls the user's server for readiness.

**Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model_id` | `str` | Yes | — | any-guardrail model identifier. Looked up in :data:`LLAMAFILE_ARTIFACTS` when ``repo_id``/``filename`` overrides weren't supplied to the constructor. In external-server mode this is recorded as metadata only. |

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

In external-server mode there is no subprocess to terminate and
``self.base_url`` is preserved so the provider stays reusable.

**Returns:** `None`
