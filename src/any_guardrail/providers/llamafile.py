"""Provider backed by Mozilla AI ``llamafile`` binaries.

Llamafile (https://github.com/Mozilla-Ocho/llamafile) packages a decoder LLM
(llama.cpp + GGUF weights) into a single, multi-platform executable via
Cosmopolitan Libc. This provider spawns the binary in OpenAI-compatible HTTP
server mode and routes :meth:`generate_chat` calls to
``POST /v1/chat/completions``.

Unlike :class:`EncoderfileProvider`, this provider does not implement
:meth:`pre_process`/:meth:`infer` — llamafile is a chat-style backend, so
:meth:`generate_chat` is the only inference entry point. Guardrails that touch
this provider directly (e.g. :class:`GraniteGuardian`) call ``generate_chat``;
they do not call ``infer``.
"""

from __future__ import annotations

import atexit
import json
import os
import socket
import stat
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from any_guardrail.providers._llamafile_artifacts import resolve_artifact
from any_guardrail.providers.base import Provider
from any_guardrail.types import (
    AnyDict,
    GuardrailInferenceOutput,
    GuardrailPreprocessOutput,
)

try:
    from huggingface_hub import hf_hub_download

    MISSING_PACKAGES_ERROR: ImportError | None = None
except ImportError as e:
    MISSING_PACKAGES_ERROR = e


_STARTUP_POLL_INTERVAL = 0.5


def _free_port() -> int:
    """Ask the kernel for a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class LlamafileProvider(Provider[AnyDict, AnyDict]):
    """Run inference through a local ``llamafile`` binary's HTTP server.

    The provider spawns the binary as a subprocess listening on ``--host``/
    ``--port`` (server mode is implicit when a port is given in llamafile
    0.10+), with ``--no-webui`` to suppress the UI, then polls ``GET /health``
    for readiness and issues ``POST /v1/chat/completions`` calls. Output is
    normalized to the same shape :meth:`HuggingFaceProvider.generate_chat`
    returns so guardrails are provider-agnostic.

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

    """

    def __init__(
        self,
        binary_path: str | None = None,
        repo_id: str | None = None,
        filename: str | None = None,
        port: int | None = None,
        host: str = "127.0.0.1",
        startup_timeout: float = 120.0,
        request_timeout: float = 120.0,
        cache_dir: str | None = None,
        n_gpu_layers: int | None = None,
        context_size: int | None = None,
        extra_args: list[str] | None = None,
    ) -> None:
        """Initialize the llamafile provider."""
        if binary_path is None and MISSING_PACKAGES_ERROR is not None:
            msg = (
                "Missing packages for LlamafileProvider auto-download. "
                "You can try `pip install 'any-guardrail[llamafile]'`, or pass a local "
                "`binary_path=` if you already have a llamafile on disk."
            )
            raise ImportError(msg) from MISSING_PACKAGES_ERROR

        if (repo_id is None) != (filename is None):
            msg = "repo_id and filename must be provided together (both or neither)."
            raise ValueError(msg)

        self.binary_path = binary_path
        self.repo_id_override = repo_id
        self.filename_override = filename
        self.port = port
        self.host = host
        self.startup_timeout = startup_timeout
        self.request_timeout = request_timeout
        self.cache_dir = cache_dir
        self.n_gpu_layers = n_gpu_layers
        self.context_size = context_size
        self.extra_args = list(extra_args) if extra_args else []

        self.process: subprocess.Popen[bytes] | None = None
        self.base_url: str | None = None
        self.model_id: str | None = None
        # ``model`` is kept for parity with HuggingFaceProvider so callers that
        # introspect ``provider.model`` don't crash.
        self.model: AnyDict = {}

        atexit.register(self.close)

    def _resolve_binary(self, model_id: str) -> Path:
        """Return a usable binary path for ``model_id``."""
        if self.binary_path is not None:
            return Path(self.binary_path)

        if self.repo_id_override is not None and self.filename_override is not None:
            repo_id, filename = self.repo_id_override, self.filename_override
        else:
            repo_id, filename = resolve_artifact(model_id)

        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=self.cache_dir,
        )
        return Path(downloaded)

    def _ensure_executable(self, path: Path) -> None:
        """Make sure the binary at ``path`` has the owner-executable bit set.

        Only the owner-execute bit is added — downloaded artifacts shouldn't be
        made world-executable on the host.
        """
        current_mode = path.stat().st_mode
        path.chmod(current_mode | stat.S_IXUSR)

    def _wait_ready(self) -> None:
        """Poll ``GET /health`` until the server returns 200 or we time out."""
        assert self.base_url is not None
        deadline = time.monotonic() + self.startup_timeout
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                msg = (
                    f"llamafile subprocess exited prematurely with code "
                    f"{self.process.returncode} before becoming ready."
                )
                raise RuntimeError(msg)
            try:
                request = urllib.request.Request(  # noqa: S310 - URL targets the llamafile subprocess this provider spawned
                    f"{self.base_url}/health",
                    method="GET",
                )
                with urllib.request.urlopen(request, timeout=2.0) as resp:  # noqa: S310 - URL targets the llamafile subprocess this provider spawned
                    if resp.status == 200:
                        return
            except (urllib.error.URLError, ConnectionError, OSError) as e:
                last_error = e
            time.sleep(_STARTUP_POLL_INTERVAL)
        msg = (
            f"llamafile server at {self.base_url} did not become ready within "
            f"{self.startup_timeout}s. Last error: {last_error!r}"
        )
        raise TimeoutError(msg)

    def load_model(self, model_id: str, **kwargs: Any) -> None:
        """Resolve the llamafile binary for ``model_id`` and start its HTTP server.

        Args:
            model_id: any-guardrail model identifier. Looked up in
                :data:`LLAMAFILE_ARTIFACTS` when ``repo_id``/``filename``
                overrides weren't supplied to the constructor.
            **kwargs: Ignored. Present to match the Provider signature.

        """
        del kwargs

        # If a previous binary is still running, tear it down first so we don't
        # leak the subprocess and its port.
        if self.process is not None:
            self.close()

        binary = self._resolve_binary(model_id)
        if not binary.exists():
            msg = f"llamafile binary not found at {binary}"
            raise FileNotFoundError(msg)
        self._ensure_executable(binary)

        port = self.port or _free_port()
        # Llamafiles are Cosmopolitan APE polyglots whose magic bytes (``MZ``)
        # macOS arm64 won't exec directly. The APE prelude is valid POSIX shell
        # that exec's into the binary, so launching via ``sh`` works portably
        # across Linux and macOS (the shell exec's, so our Popen PID is still
        # the llama server PID — terminate() works).
        # Server mode is implicit in llamafile 0.10+ when a port is given;
        # ``--jinja`` is also the default but kept explicit for clarity.
        cmd: list[str] = [
            "sh",
            str(binary),
            "--host",
            self.host,
            "--port",
            str(port),
            "--no-webui",
            "--jinja",
        ]
        if self.n_gpu_layers is not None:
            cmd.extend(["--n-gpu-layers", str(self.n_gpu_layers)])
        if self.context_size is not None:
            cmd.extend(["--ctx-size", str(self.context_size)])
        cmd.extend(self.extra_args)

        stdout: Any = subprocess.DEVNULL if not os.environ.get("LLAMAFILE_VERBOSE") else None
        self.process = subprocess.Popen(cmd, stdout=stdout, stderr=stdout)  # noqa: S603 - cmd built from trusted parts
        self.base_url = f"http://{self.host}:{port}"
        self.model_id = model_id
        self.model = {"binary_path": str(binary), "model_id": model_id}

        try:
            self._wait_ready()
        except Exception:
            self.close()
            raise

    def pre_process(self, *args: Any, **kwargs: Any) -> GuardrailPreprocessOutput[AnyDict]:
        """Not supported — llamafile is a chat-style backend.

        Use :meth:`generate_chat` instead. Decoder-LLM guardrails like
        :class:`GraniteGuardian` route through ``generate_chat`` automatically.
        """
        del args, kwargs
        msg = "LlamafileProvider does not implement pre_process(); use generate_chat() instead."
        raise NotImplementedError(msg)

    def infer(self, model_inputs: GuardrailPreprocessOutput[AnyDict]) -> GuardrailInferenceOutput[AnyDict]:
        """Not supported — llamafile is a chat-style backend.

        Use :meth:`generate_chat` instead.
        """
        del model_inputs
        msg = "LlamafileProvider does not implement infer(); use generate_chat() instead."
        raise NotImplementedError(msg)

    def generate_chat(
        self,
        messages: list[AnyDict],
        *,
        max_new_tokens: int,
        do_sample: bool = False,
        temperature: float | None = None,
        chat_template_kwargs: AnyDict | None = None,
        generation_kwargs: AnyDict | None = None,
    ) -> GuardrailInferenceOutput[AnyDict]:
        """POST messages to ``/v1/chat/completions`` and return the generated text.

        Maps the unified ``generate_chat`` contract onto OpenAI-shape semantics:
        ``do_sample=False`` becomes ``temperature=0`` (greedy); ``do_sample=True``
        forwards ``temperature``. ``chat_template_kwargs`` is added to the request
        body only when non-empty — relies on llama.cpp's optional
        ``chat_template_kwargs`` extension on the chat-completions endpoint.

        ``prompt_token_count`` and ``completion_token_count`` are populated from
        the response's ``usage`` block when present.
        """
        if self.base_url is None:
            msg = "load_model() must be called before generate_chat()"
            raise RuntimeError(msg)

        body: AnyDict = {
            "messages": messages,
            "max_tokens": max_new_tokens,
            "stream": False,
        }
        if do_sample:
            # Sampling: only forward an explicit temperature; otherwise let the
            # llamafile server use its own default (don't silently collapse to
            # greedy by sending temperature=0).
            if temperature is not None:
                body["temperature"] = temperature
        else:
            # Greedy: pin temperature to 0 so the server doesn't sample.
            body["temperature"] = 0
        if chat_template_kwargs:
            body["chat_template_kwargs"] = chat_template_kwargs
        if generation_kwargs:
            body.update(generation_kwargs)

        payload = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(  # noqa: S310 - URL targets the llamafile subprocess this provider spawned
            f"{self.base_url}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.request_timeout) as resp:  # noqa: S310 - URL targets the llamafile subprocess this provider spawned
            response_body = resp.read()
        parsed = json.loads(response_body)
        generated_text = str(parsed["choices"][0]["message"]["content"])
        usage = parsed.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")

        return GuardrailInferenceOutput(
            data={
                "generated_text": generated_text,
                "prompt_token_count": int(prompt_tokens) if prompt_tokens is not None else None,
                "completion_token_count": int(completion_tokens) if completion_tokens is not None else None,
                "raw": parsed,
            }
        )

    def close(self) -> None:
        """Terminate the llamafile subprocess. Idempotent."""
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.process = None
        self.base_url = None

    def __del__(self) -> None:
        """Best-effort cleanup on GC. ``atexit`` also covers process exit."""
        try:
            self.close()
        except Exception:  # noqa: S110 - __del__ must never raise
            pass
