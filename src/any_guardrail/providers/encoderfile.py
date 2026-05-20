"""Provider backed by Mozilla AI ``encoderfile`` binaries.

Encoderfile (https://github.com/mozilla-ai/encoderfile) packages a transformer
encoder + classification head into a single, self-contained executable. This
provider runs the binary in HTTP server mode and proxies inference calls over
``localhost``.
"""

from __future__ import annotations

import atexit
import json
import os
import platform
import socket
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from any_guardrail.providers._encoderfile_artifacts import resolve_artifact_path
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


_DEFAULT_ENCODERFILE_REPO = "mozilla-ai/encoderfile"
_STARTUP_POLL_INTERVAL = 0.25


def _detect_platform_tag() -> str:
    """Return the platform tag used in encoderfile artifact filenames.

    Format: ``{arch}-{vendor}-{os}-{libc}``, e.g. ``aarch64-apple-darwin``.
    """
    machine = platform.machine().lower()
    arch_map = {
        "arm64": "aarch64",
        "aarch64": "aarch64",
        "x86_64": "x86_64",
        "amd64": "x86_64",
    }
    if machine not in arch_map:
        msg = f"Unsupported CPU architecture for encoderfile: {platform.machine()!r}"
        raise RuntimeError(msg)
    arch = arch_map[machine]

    if sys.platform == "darwin":
        os_tag = "apple-darwin"
    elif sys.platform.startswith("linux"):
        os_tag = "linux-gnu"
    else:
        msg = f"Unsupported platform for encoderfile: {sys.platform!r} (only macOS and Linux are supported)"
        raise RuntimeError(msg)
    return f"{arch}-{os_tag}"


def _free_port() -> int:
    """Ask the kernel for a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class EncoderfileProvider(Provider[AnyDict, AnyDict]):
    """Run inference through a local ``encoderfile`` binary's HTTP server.

    The provider spawns the binary as a subprocess, polls for readiness, then
    issues ``POST /predict`` calls. Output is normalized to the same shape
    HuggingFaceProvider returns so downstream guardrails are provider-agnostic.

    Args:
        binary_path: Path to a pre-built ``.encoderfile``. If omitted, the
            platform-appropriate artifact is auto-downloaded from
            ``mozilla-ai/encoderfile`` using the model_id passed to
            ``load_model``.
        port: TCP port to bind the encoderfile HTTP server. Defaults to a
            kernel-chosen free port.
        host: Bind address. Defaults to ``"127.0.0.1"``.
        startup_timeout: Seconds to wait for the server to become ready.
        request_timeout: Per-request timeout for ``/predict`` calls.
        cache_dir: Directory passed to ``hf_hub_download`` for auto-downloaded
            binaries.
        encoderfile_repo: Override the source HF repo. Defaults to
            ``mozilla-ai/encoderfile``.

    """

    def __init__(
        self,
        binary_path: str | None = None,
        port: int | None = None,
        host: str = "127.0.0.1",
        startup_timeout: float = 60.0,
        request_timeout: float = 60.0,
        cache_dir: str | None = None,
        encoderfile_repo: str = _DEFAULT_ENCODERFILE_REPO,
    ) -> None:
        """Initialize the encoderfile provider."""
        if binary_path is None and MISSING_PACKAGES_ERROR is not None:
            msg = (
                "Missing packages for EncoderfileProvider auto-download. "
                "You can try `pip install 'any-guardrail[encoderfile]'`, or pass a local "
                "`binary_path=` if you already have a built encoderfile."
            )
            raise ImportError(msg) from MISSING_PACKAGES_ERROR

        self.binary_path = binary_path
        self.port = port
        self.host = host
        self.startup_timeout = startup_timeout
        self.request_timeout = request_timeout
        self.cache_dir = cache_dir
        self.encoderfile_repo = encoderfile_repo

        self.process: subprocess.Popen[bytes] | None = None
        self.base_url: str | None = None
        self.model_id: str | None = None
        # ``model`` is kept for parity with HuggingFaceProvider (some callers
        # access ``provider.model`` for compatibility). It carries the binary
        # path and a labels list when known.
        self.model: AnyDict = {}

        atexit.register(self.close)

    def _resolve_binary(self, model_id: str) -> Path:
        """Return a usable, executable binary path for ``model_id``."""
        if self.binary_path is not None:
            return Path(self.binary_path)

        platform_tag = _detect_platform_tag()
        artifact_path = resolve_artifact_path(model_id, platform_tag)
        downloaded = hf_hub_download(
            repo_id=self.encoderfile_repo,
            filename=artifact_path,
            cache_dir=self.cache_dir,
        )
        return Path(downloaded)

    def _ensure_executable(self, path: Path) -> None:
        """Make sure the binary at ``path`` has the user-executable bit set."""
        current_mode = path.stat().st_mode
        path.chmod(current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    def _wait_ready(self) -> None:
        """Poll ``/predict`` with a tiny input until the server responds or times out."""
        assert self.base_url is not None
        deadline = time.monotonic() + self.startup_timeout
        probe_payload = json.dumps({"inputs": ["ready?"]}).encode("utf-8")
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                msg = (
                    f"encoderfile subprocess exited prematurely with code "
                    f"{self.process.returncode} before becoming ready."
                )
                raise RuntimeError(msg)
            try:
                request = urllib.request.Request(  # noqa: S310 - localhost only
                    f"{self.base_url}/predict",
                    data=probe_payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=2.0) as resp:  # noqa: S310 - localhost only
                    if 200 <= resp.status < 500:
                        return
            except (urllib.error.URLError, ConnectionError, OSError) as e:
                last_error = e
            time.sleep(_STARTUP_POLL_INTERVAL)
        msg = (
            f"encoderfile server at {self.base_url} did not become ready within "
            f"{self.startup_timeout}s. Last error: {last_error!r}"
        )
        raise TimeoutError(msg)

    def load_model(self, model_id: str, **kwargs: Any) -> None:
        """Load the encoderfile binary for ``model_id`` and start its HTTP server.

        Args:
            model_id: any-guardrail model identifier. Used to look up the right
                encoderfile artifact when auto-downloading.
            **kwargs: Ignored. Present to match the Provider signature.

        """
        del kwargs  # not used; the binary owns all model state.

        binary = self._resolve_binary(model_id)
        if not binary.exists():
            msg = f"encoderfile binary not found at {binary}"
            raise FileNotFoundError(msg)
        self._ensure_executable(binary)

        port = self.port or _free_port()
        cmd = [
            str(binary),
            "serve",
            "--http-hostname",
            self.host,
            "--http-port",
            str(port),
            "--disable-grpc",
        ]
        # stdout/stderr discarded; encoderfile is chatty on startup. Users who
        # need to debug can set ENCODERFILE_VERBOSE=1.
        stdout: Any = subprocess.DEVNULL if not os.environ.get("ENCODERFILE_VERBOSE") else None
        self.process = subprocess.Popen(cmd, stdout=stdout, stderr=stdout)  # noqa: S603 - cmd built from trusted parts
        self.base_url = f"http://{self.host}:{port}"
        self.model_id = model_id
        self.model = {"binary_path": str(binary), "model_id": model_id}

        try:
            self._wait_ready()
        except Exception:
            self.close()
            raise

    def pre_process(self, input_text: str | list[str], **kwargs: Any) -> GuardrailPreprocessOutput[AnyDict]:
        """Wrap raw text into the encoderfile request body.

        Encoderfile does its own tokenization inside the binary; the only
        client-side preparation is shaping the JSON payload.
        """
        del kwargs  # encoderfile has no per-request tokenization knobs to forward.
        texts = [input_text] if isinstance(input_text, str) else list(input_text)
        return GuardrailPreprocessOutput(data={"inputs": texts})

    def infer(self, model_inputs: GuardrailPreprocessOutput[AnyDict]) -> GuardrailInferenceOutput[AnyDict]:
        """POST the preprocessed payload to the running encoderfile server.

        Returns the same uniform shape as HuggingFaceProvider: ``logits``,
        ``scores``, ``predicted_indices``, ``predicted_labels``.
        """
        if self.base_url is None:
            msg = "load_model() must be called before infer()"
            raise RuntimeError(msg)

        payload = json.dumps(model_inputs.data).encode("utf-8")
        request = urllib.request.Request(  # noqa: S310 - localhost only
            f"{self.base_url}/predict",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.request_timeout) as resp:  # noqa: S310 - localhost only
            body = resp.read()
        parsed = json.loads(body)
        results = parsed["results"]
        logits = np.array([r["logits"] for r in results], dtype=np.float32)
        scores = np.array([r["scores"] for r in results], dtype=np.float32)
        predicted_indices = [int(r["predicted_index"]) for r in results]
        predicted_labels = [str(r["predicted_label"]) for r in results]
        return GuardrailInferenceOutput(
            data={
                "logits": logits,
                "scores": scores,
                "predicted_indices": predicted_indices,
                "predicted_labels": predicted_labels,
            }
        )

    def close(self) -> None:
        """Terminate the encoderfile subprocess. Idempotent."""
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
