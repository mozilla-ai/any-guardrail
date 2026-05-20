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
