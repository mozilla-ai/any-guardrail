import json
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from any_guardrail.providers.encoderfile import EncoderfileProvider, _detect_platform_tag
from any_guardrail.types import GuardrailInferenceOutput, GuardrailPreprocessOutput


def _stub_response(payload: dict[str, Any]) -> MagicMock:
    """Build a context-manager that mimics urlopen()'s response object."""
    response = MagicMock()
    response.read.return_value = json.dumps(payload).encode("utf-8")
    response.status = 200
    response.__enter__ = MagicMock(return_value=response)
    response.__exit__ = MagicMock(return_value=False)
    return response


@pytest.fixture
def fake_subprocess() -> Any:
    """Patch subprocess.Popen so load_model doesn't actually run a binary."""
    with patch("any_guardrail.providers.encoderfile.subprocess.Popen") as mock_popen:
        proc = MagicMock()
        proc.poll.return_value = None  # still running
        mock_popen.return_value = proc
        yield mock_popen, proc


@pytest.fixture
def fake_ready_probe() -> Any:
    """Make the readiness probe succeed immediately."""
    with patch("any_guardrail.providers.encoderfile.urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value = _stub_response({"results": [], "model_id": "stub"})
        yield mock_urlopen


def test_pre_process_wraps_string() -> None:
    provider = EncoderfileProvider(binary_path="/fake")
    result = provider.pre_process("hello")
    assert isinstance(result, GuardrailPreprocessOutput)
    assert result.data == {"inputs": ["hello"]}


def test_pre_process_wraps_list() -> None:
    provider = EncoderfileProvider(binary_path="/fake")
    result = provider.pre_process(["one", "two", "three"])
    assert result.data == {"inputs": ["one", "two", "three"]}


def test_detect_platform_tag_macos_arm64() -> None:
    with (
        patch("any_guardrail.providers.encoderfile.platform.machine", return_value="arm64"),
        patch("any_guardrail.providers.encoderfile.sys.platform", "darwin"),
    ):
        assert _detect_platform_tag() == "aarch64-apple-darwin"


def test_detect_platform_tag_linux_x86_64() -> None:
    with (
        patch("any_guardrail.providers.encoderfile.platform.machine", return_value="x86_64"),
        patch("any_guardrail.providers.encoderfile.sys.platform", "linux"),
    ):
        assert _detect_platform_tag() == "x86_64-linux-gnu"


def test_detect_platform_tag_unsupported_platform_raises() -> None:
    with (
        patch("any_guardrail.providers.encoderfile.platform.machine", return_value="x86_64"),
        patch("any_guardrail.providers.encoderfile.sys.platform", "win32"),
        pytest.raises(RuntimeError, match="Unsupported platform"),
    ):
        _detect_platform_tag()


def test_detect_platform_tag_unsupported_arch_raises() -> None:
    with (
        patch("any_guardrail.providers.encoderfile.platform.machine", return_value="ppc64le"),
        patch("any_guardrail.providers.encoderfile.sys.platform", "linux"),
        pytest.raises(RuntimeError, match="Unsupported CPU architecture"),
    ):
        _detect_platform_tag()


def test_load_model_uses_binary_path(tmp_path: Any, fake_subprocess: Any, fake_ready_probe: Any) -> None:
    binary = tmp_path / "fake.encoderfile"
    binary.write_bytes(b"#!/bin/sh\necho stub\n")

    mock_popen, _ = fake_subprocess
    provider = EncoderfileProvider(binary_path=str(binary), port=12345)
    provider.load_model("ProtectAI/deberta-v3-base-prompt-injection-v2")

    args, _ = mock_popen.call_args
    cmd = args[0]
    assert cmd[0] == str(binary)
    assert "serve" in cmd
    assert "--http-port" in cmd
    assert "12345" in cmd
    assert "--disable-grpc" in cmd
    assert provider.base_url == "http://127.0.0.1:12345"


def test_load_model_auto_downloads_correct_artifact(tmp_path: Any, fake_subprocess: Any, fake_ready_probe: Any) -> None:
    binary = tmp_path / "downloaded.encoderfile"
    binary.write_bytes(b"#!/bin/sh\necho stub\n")

    with (
        patch(
            "any_guardrail.providers.encoderfile.hf_hub_download",
            return_value=str(binary),
        ) as mock_download,
        patch("any_guardrail.providers.encoderfile._detect_platform_tag", return_value="aarch64-apple-darwin"),
    ):
        provider = EncoderfileProvider(port=12346)
        provider.load_model("ProtectAI/deberta-v3-base-prompt-injection-v2")

    _, download_kwargs = mock_download.call_args
    assert download_kwargs["repo_id"] == "mozilla-ai/encoderfile"
    assert download_kwargs["filename"] == (
        "protectai/deberta-v3-base-prompt-injection-v2/"
        "deberta-v3-base-prompt-injection-v2.aarch64-apple-darwin.encoderfile"
    )


def test_load_model_unknown_model_id_raises(tmp_path: Any) -> None:
    provider = EncoderfileProvider(port=12347)
    with pytest.raises(KeyError, match="No encoderfile artifact registered"):
        provider.load_model("unknown/model")


def test_infer_parses_response(tmp_path: Any, fake_subprocess: Any) -> None:
    binary = tmp_path / "fake.encoderfile"
    binary.write_bytes(b"#!/bin/sh\necho stub\n")

    server_response = {
        "results": [
            {
                "logits": [0.1, 0.9],
                "scores": [0.27, 0.73],
                "predicted_index": 1,
                "predicted_label": "INJECTION",
            },
            {
                "logits": [0.95, 0.05],
                "scores": [0.95, 0.05],
                "predicted_index": 0,
                "predicted_label": "SAFE",
            },
        ],
        "model_id": "stub",
    }

    # First urlopen() call is the readiness probe; second is the real infer.
    with patch("any_guardrail.providers.encoderfile.urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.side_effect = [
            _stub_response({"results": [], "model_id": "stub"}),
            _stub_response(server_response),
        ]
        provider = EncoderfileProvider(binary_path=str(binary), port=12348)
        provider.load_model("ProtectAI/deberta-v3-base-prompt-injection-v2")

        result = provider.infer(GuardrailPreprocessOutput(data={"inputs": ["a", "b"]}))

    assert isinstance(result, GuardrailInferenceOutput)
    np.testing.assert_array_almost_equal(result.data["logits"], np.array([[0.1, 0.9], [0.95, 0.05]]))
    np.testing.assert_array_almost_equal(result.data["scores"], np.array([[0.27, 0.73], [0.95, 0.05]]))
    assert result.data["predicted_indices"] == [1, 0]
    assert result.data["predicted_labels"] == ["INJECTION", "SAFE"]


def test_infer_before_load_model_raises() -> None:
    provider = EncoderfileProvider(binary_path="/fake")
    with pytest.raises(RuntimeError, match="load_model"):
        provider.infer(GuardrailPreprocessOutput(data={"inputs": ["x"]}))


def test_close_terminates_process(tmp_path: Any, fake_subprocess: Any, fake_ready_probe: Any) -> None:
    binary = tmp_path / "fake.encoderfile"
    binary.write_bytes(b"#!/bin/sh\necho stub\n")
    _, proc = fake_subprocess

    provider = EncoderfileProvider(binary_path=str(binary), port=12349)
    provider.load_model("ProtectAI/deberta-v3-base-prompt-injection-v2")
    provider.close()

    proc.terminate.assert_called_once()
    assert provider.process is None
    assert provider.base_url is None


def test_close_is_idempotent() -> None:
    provider = EncoderfileProvider(binary_path="/fake")
    provider.close()
    provider.close()


def test_load_model_makes_binary_executable(tmp_path: Any, fake_subprocess: Any, fake_ready_probe: Any) -> None:
    binary = tmp_path / "not_executable.encoderfile"
    binary.write_bytes(b"#!/bin/sh\necho stub\n")
    binary.chmod(0o644)  # explicitly remove exec bit

    provider = EncoderfileProvider(binary_path=str(binary), port=12350)
    provider.load_model("ProtectAI/deberta-v3-base-prompt-injection-v2")

    assert binary.stat().st_mode & 0o111, "expected at least one execute bit set after load_model"


def test_context_manager_returns_self_from_enter() -> None:
    """`with EncoderfileProvider() as p:` should bind `p` to the provider instance."""
    provider = EncoderfileProvider(binary_path="/fake")
    with provider as p:
        assert p is provider


def test_context_manager_calls_close_on_exit(tmp_path: Any, fake_subprocess: Any, fake_ready_probe: Any) -> None:
    """Exiting the `with` block should terminate the subprocess."""
    binary = tmp_path / "fake.encoderfile"
    binary.write_bytes(b"#!/bin/sh\necho stub\n")
    _, proc = fake_subprocess

    with EncoderfileProvider(binary_path=str(binary), port=12351) as provider:
        provider.load_model("ProtectAI/deberta-v3-base-prompt-injection-v2")
        assert provider.process is not None  # spawned

    proc.terminate.assert_called_once()
    assert provider.process is None
    assert provider.base_url is None


def test_context_manager_calls_close_even_when_block_raises(
    tmp_path: Any, fake_subprocess: Any, fake_ready_probe: Any
) -> None:
    """Exceptions inside the `with` block must still terminate the subprocess."""
    binary = tmp_path / "fake.encoderfile"
    binary.write_bytes(b"#!/bin/sh\necho stub\n")
    _, proc = fake_subprocess

    provider = EncoderfileProvider(binary_path=str(binary), port=12352)
    with pytest.raises(RuntimeError, match="boom"), provider:
        provider.load_model("ProtectAI/deberta-v3-base-prompt-injection-v2")
        raise RuntimeError("boom")

    proc.terminate.assert_called_once()
    assert provider.process is None


def test_load_model_retries_on_bind_race_when_port_auto_picked(tmp_path: Any) -> None:
    """When we auto-pick the port and the subprocess exits prematurely, retry with a fresh port.

    Simulates the TOCTOU race documented in the `_BIND_RACE_RETRIES` docstring:
    the first spawned subprocess dies before becoming ready (returncode 1),
    the second succeeds. `load_model` should retry transparently.
    """
    binary = tmp_path / "fake.encoderfile"
    binary.write_bytes(b"#!/bin/sh\necho stub\n")

    # First proc dies; second proc stays alive.
    dead_proc = MagicMock()
    dead_proc.poll.return_value = 1  # subprocess already exited
    dead_proc.returncode = 1

    live_proc = MagicMock()
    live_proc.poll.return_value = None  # still running

    with (
        patch("any_guardrail.providers.encoderfile.subprocess.Popen") as mock_popen,
        patch("any_guardrail.providers.encoderfile.urllib.request.urlopen") as mock_urlopen,
    ):
        mock_popen.side_effect = [dead_proc, live_proc]
        mock_urlopen.return_value = _stub_response({"results": [], "model_id": "stub"})

        # No `port=` kwarg => auto-pick => retry path is active.
        provider = EncoderfileProvider(binary_path=str(binary), startup_timeout=2.0)
        provider.load_model("ProtectAI/deberta-v3-base-prompt-injection-v2")

    assert mock_popen.call_count == 2, "expected a second Popen after the first subprocess died"
    assert provider.process is live_proc
    # The first (dead) proc was terminated/closed as part of the retry path.
    dead_proc.terminate.assert_not_called()  # `close()` skips terminate when poll()!=None
    provider.close()


def test_load_model_does_not_retry_when_port_pinned(tmp_path: Any) -> None:
    """If the user pinned a port, a bind failure is their config problem — surface it, don't retry."""
    binary = tmp_path / "fake.encoderfile"
    binary.write_bytes(b"#!/bin/sh\necho stub\n")

    dead_proc = MagicMock()
    dead_proc.poll.return_value = 1
    dead_proc.returncode = 1

    with (
        patch("any_guardrail.providers.encoderfile.subprocess.Popen") as mock_popen,
        patch("any_guardrail.providers.encoderfile.urllib.request.urlopen") as mock_urlopen,
    ):
        mock_popen.return_value = dead_proc
        mock_urlopen.side_effect = ConnectionRefusedError("nothing listening")

        provider = EncoderfileProvider(binary_path=str(binary), port=12353, startup_timeout=2.0)
        with pytest.raises(RuntimeError, match="exited prematurely"):
            provider.load_model("ProtectAI/deberta-v3-base-prompt-injection-v2")

    assert mock_popen.call_count == 1, "expected exactly one Popen attempt when port is user-pinned"


def test_load_model_gives_up_after_max_bind_race_retries(tmp_path: Any) -> None:
    """If every retry attempt also dies, the final attempt's error is surfaced to the caller."""
    binary = tmp_path / "fake.encoderfile"
    binary.write_bytes(b"#!/bin/sh\necho stub\n")

    dead_proc = MagicMock()
    dead_proc.poll.return_value = 1
    dead_proc.returncode = 1

    with (
        patch("any_guardrail.providers.encoderfile.subprocess.Popen") as mock_popen,
        patch("any_guardrail.providers.encoderfile.urllib.request.urlopen") as mock_urlopen,
    ):
        mock_popen.return_value = dead_proc
        mock_urlopen.side_effect = ConnectionRefusedError("nothing listening")

        provider = EncoderfileProvider(binary_path=str(binary), startup_timeout=2.0)
        with pytest.raises(RuntimeError, match="exited prematurely"):
            provider.load_model("ProtectAI/deberta-v3-base-prompt-injection-v2")

    assert mock_popen.call_count == EncoderfileProvider._BIND_RACE_RETRIES
