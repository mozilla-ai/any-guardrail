import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from any_guardrail.providers.llamafile import LlamafileProvider


def _stub_response(payload: dict[str, Any], status: int = 200) -> MagicMock:
    """Build a context-manager that mimics urlopen()'s response object."""
    response = MagicMock()
    response.read.return_value = json.dumps(payload).encode("utf-8")
    response.status = status
    response.__enter__ = MagicMock(return_value=response)
    response.__exit__ = MagicMock(return_value=False)
    return response


@pytest.fixture
def fake_subprocess() -> Any:
    """Patch subprocess.Popen so load_model doesn't actually run a binary."""
    with patch("any_guardrail.providers.llamafile.subprocess.Popen") as mock_popen:
        proc = MagicMock()
        proc.poll.return_value = None  # still running
        mock_popen.return_value = proc
        yield mock_popen, proc


@pytest.fixture
def fake_ready_probe() -> Any:
    """Make the readiness probe succeed immediately."""
    with patch("any_guardrail.providers.llamafile.urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value = _stub_response({"status": "ok"})
        yield mock_urlopen


def test_constructor_rejects_partial_repo_filename_override() -> None:
    """Either both repo_id and filename or neither — not one of the two."""
    with pytest.raises(ValueError, match="must be provided together"):
        LlamafileProvider(repo_id="some/repo")
    with pytest.raises(ValueError, match="must be provided together"):
        LlamafileProvider(filename="some.llamafile")


def test_load_model_uses_binary_path(tmp_path: Any, fake_subprocess: Any, fake_ready_probe: Any) -> None:
    binary = tmp_path / "fake.llamafile"
    binary.write_bytes(b"#!/bin/sh\necho stub\n")

    mock_popen, _ = fake_subprocess
    provider = LlamafileProvider(binary_path=str(binary), port=23456)
    provider.load_model("ibm-granite/granite-guardian-4.1-8b")

    args, _ = mock_popen.call_args
    cmd = args[0]
    # Cosmopolitan APE binaries are launched via ``sh`` so the shell prelude
    # can bootstrap the actual binary (macOS arm64 can't exec MZ directly).
    assert cmd[0] == "sh"
    assert cmd[1] == str(binary)
    assert "--no-webui" in cmd
    assert "--jinja" in cmd
    assert "--port" in cmd
    assert "23456" in cmd
    assert provider.base_url == "http://127.0.0.1:23456"
    provider.close()


def test_load_model_auto_downloads_from_artifact_map(
    tmp_path: Any, fake_subprocess: Any, fake_ready_probe: Any
) -> None:
    binary = tmp_path / "downloaded.llamafile"
    binary.write_bytes(b"#!/bin/sh\necho stub\n")

    with patch(
        "any_guardrail.providers.llamafile.hf_hub_download",
        return_value=str(binary),
    ) as mock_download:
        provider = LlamafileProvider(port=23457)
        provider.load_model("ibm-granite/granite-guardian-4.1-8b")

    _, download_kwargs = mock_download.call_args
    assert download_kwargs["repo_id"] == "mozilla-ai/llamafile_0.10_alpha"
    assert download_kwargs["filename"] == "granite-guardian-4.1-8b.Q6_K.llamafile"
    provider.close()


def test_load_model_with_repo_id_filename_override(tmp_path: Any, fake_subprocess: Any, fake_ready_probe: Any) -> None:
    binary = tmp_path / "override.llamafile"
    binary.write_bytes(b"#!/bin/sh\necho stub\n")

    with patch(
        "any_guardrail.providers.llamafile.hf_hub_download",
        return_value=str(binary),
    ) as mock_download:
        provider = LlamafileProvider(
            repo_id="custom/repo",
            filename="my-model.llamafile",
            port=23458,
        )
        # model_id is unmapped — that's fine because the constructor override wins.
        provider.load_model("custom/model-id")

    _, download_kwargs = mock_download.call_args
    assert download_kwargs["repo_id"] == "custom/repo"
    assert download_kwargs["filename"] == "my-model.llamafile"
    provider.close()


def test_load_model_unknown_model_id_raises_keyerror() -> None:
    provider = LlamafileProvider(port=23459)
    with pytest.raises(KeyError, match="No llamafile artifact registered"):
        provider.load_model("unmapped/model")


def test_load_model_passes_n_gpu_layers_and_ctx_size(
    tmp_path: Any, fake_subprocess: Any, fake_ready_probe: Any
) -> None:
    binary = tmp_path / "fake.llamafile"
    binary.write_bytes(b"#!/bin/sh\necho stub\n")
    mock_popen, _ = fake_subprocess

    provider = LlamafileProvider(
        binary_path=str(binary),
        port=23460,
        n_gpu_layers=32,
        context_size=4096,
        extra_args=["--api-key", "secret"],
    )
    provider.load_model("ibm-granite/granite-guardian-4.1-8b")

    cmd = mock_popen.call_args.args[0]
    assert "--n-gpu-layers" in cmd
    assert cmd[cmd.index("--n-gpu-layers") + 1] == "32"
    assert "--ctx-size" in cmd
    assert cmd[cmd.index("--ctx-size") + 1] == "4096"
    assert cmd[-2:] == ["--api-key", "secret"]
    provider.close()


def test_load_model_closes_previous_subprocess(tmp_path: Any, fake_subprocess: Any, fake_ready_probe: Any) -> None:
    binary = tmp_path / "fake.llamafile"
    binary.write_bytes(b"#!/bin/sh\necho stub\n")
    mock_popen, _ = fake_subprocess

    provider = LlamafileProvider(binary_path=str(binary), port=23461)
    provider.load_model("ibm-granite/granite-guardian-4.1-8b")
    first_proc = mock_popen.return_value

    # Loading again should terminate the first subprocess before spawning a new one.
    provider.load_model("ibm-granite/granite-guardian-4.1-8b")

    first_proc.terminate.assert_called_once()
    assert mock_popen.call_count == 2
    provider.close()


def test_load_model_makes_binary_executable(tmp_path: Any, fake_subprocess: Any, fake_ready_probe: Any) -> None:
    binary = tmp_path / "not_executable.llamafile"
    binary.write_bytes(b"#!/bin/sh\necho stub\n")
    binary.chmod(0o644)

    provider = LlamafileProvider(binary_path=str(binary), port=23462)
    provider.load_model("ibm-granite/granite-guardian-4.1-8b")

    assert binary.stat().st_mode & 0o100, "expected the owner-execute bit to be set after load_model"
    provider.close()


def test_wait_ready_polls_health_endpoint(tmp_path: Any, fake_subprocess: Any) -> None:
    """The readiness probe targets ``/health`` and POSTs nothing (GET)."""
    binary = tmp_path / "fake.llamafile"
    binary.write_bytes(b"#!/bin/sh\necho stub\n")

    with patch("any_guardrail.providers.llamafile.urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value = _stub_response({"status": "ok"})
        provider = LlamafileProvider(binary_path=str(binary), port=23463)
        provider.load_model("ibm-granite/granite-guardian-4.1-8b")

    request_arg = mock_urlopen.call_args.args[0]
    assert request_arg.full_url.endswith("/health")
    assert request_arg.method == "GET"
    provider.close()


def test_wait_ready_raises_if_subprocess_exits(tmp_path: Any) -> None:
    """If the subprocess dies during warm-up, _wait_ready raises RuntimeError."""
    binary = tmp_path / "fake.llamafile"
    binary.write_bytes(b"#!/bin/sh\necho stub\n")

    with (
        patch("any_guardrail.providers.llamafile.subprocess.Popen") as mock_popen,
        patch("any_guardrail.providers.llamafile.urllib.request.urlopen") as mock_urlopen,
    ):
        proc = MagicMock()
        proc.poll.return_value = 1  # already exited with non-zero
        proc.returncode = 1
        mock_popen.return_value = proc
        mock_urlopen.side_effect = ConnectionError("not yet")

        provider = LlamafileProvider(binary_path=str(binary), port=23464, startup_timeout=0.5)
        with pytest.raises(RuntimeError, match="exited prematurely"):
            provider.load_model("ibm-granite/granite-guardian-4.1-8b")


def test_generate_chat_posts_correct_payload(tmp_path: Any, fake_subprocess: Any) -> None:
    binary = tmp_path / "fake.llamafile"
    binary.write_bytes(b"#!/bin/sh\necho stub\n")

    completion_response = {
        "choices": [{"message": {"role": "assistant", "content": "<score>no</score>"}}],
        "usage": {"prompt_tokens": 42, "completion_tokens": 5},
    }
    with patch("any_guardrail.providers.llamafile.urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.side_effect = [
            _stub_response({"status": "ok"}),
            _stub_response(completion_response),
        ]
        provider = LlamafileProvider(binary_path=str(binary), port=23465)
        provider.load_model("ibm-granite/granite-guardian-4.1-8b")
        result = provider.generate_chat(
            messages=[{"role": "user", "content": "ping"}],
            max_new_tokens=48,
        )

    # Second call is the chat completion POST.
    chat_request = mock_urlopen.call_args_list[1].args[0]
    assert chat_request.full_url.endswith("/v1/chat/completions")
    assert chat_request.method == "POST"
    body = json.loads(chat_request.data)
    assert body["messages"] == [{"role": "user", "content": "ping"}]
    assert body["max_tokens"] == 48
    assert body["temperature"] == 0  # do_sample=False => greedy
    assert body["stream"] is False
    assert "chat_template_kwargs" not in body  # not included when empty

    assert result.data["generated_text"] == "<score>no</score>"
    assert result.data["prompt_token_count"] == 42
    assert result.data["completion_token_count"] == 5
    provider.close()


def test_generate_chat_includes_chat_template_kwargs_only_when_nonempty(tmp_path: Any, fake_subprocess: Any) -> None:
    binary = tmp_path / "fake.llamafile"
    binary.write_bytes(b"#!/bin/sh\necho stub\n")

    completion_response = {
        "choices": [{"message": {"content": "ok"}}],
    }
    with patch("any_guardrail.providers.llamafile.urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.side_effect = [
            _stub_response({"status": "ok"}),
            _stub_response(completion_response),
            _stub_response(completion_response),
        ]
        provider = LlamafileProvider(binary_path=str(binary), port=23466)
        provider.load_model("ibm-granite/granite-guardian-4.1-8b")

        # No chat_template_kwargs supplied => key absent from request.
        provider.generate_chat(
            messages=[{"role": "user", "content": "x"}],
            max_new_tokens=10,
        )
        body_no_kwargs = json.loads(mock_urlopen.call_args_list[1].args[0].data)
        assert "chat_template_kwargs" not in body_no_kwargs

        # Non-empty chat_template_kwargs => key present.
        provider.generate_chat(
            messages=[{"role": "user", "content": "x"}],
            max_new_tokens=10,
            chat_template_kwargs={"add_generation_prompt": False, "documents": [{"text": "d"}]},
        )
        body_with_kwargs = json.loads(mock_urlopen.call_args_list[2].args[0].data)
        assert body_with_kwargs["chat_template_kwargs"]["add_generation_prompt"] is False
        assert body_with_kwargs["chat_template_kwargs"]["documents"] == [{"text": "d"}]
    provider.close()


def test_generate_chat_forwards_temperature_when_sampling(tmp_path: Any, fake_subprocess: Any) -> None:
    binary = tmp_path / "fake.llamafile"
    binary.write_bytes(b"#!/bin/sh\necho stub\n")

    with patch("any_guardrail.providers.llamafile.urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.side_effect = [
            _stub_response({"status": "ok"}),
            _stub_response({"choices": [{"message": {"content": "x"}}]}),
        ]
        provider = LlamafileProvider(binary_path=str(binary), port=23467)
        provider.load_model("ibm-granite/granite-guardian-4.1-8b")
        provider.generate_chat(
            messages=[{"role": "user", "content": "x"}],
            max_new_tokens=10,
            do_sample=True,
            temperature=0.7,
        )

    body = json.loads(mock_urlopen.call_args_list[1].args[0].data)
    assert body["temperature"] == 0.7
    provider.close()


def test_generate_chat_omits_temperature_when_sampling_without_explicit_value(
    tmp_path: Any, fake_subprocess: Any
) -> None:
    """Sampling without an explicit temperature must not collapse to greedy (temperature=0)."""
    binary = tmp_path / "fake.llamafile"
    binary.write_bytes(b"#!/bin/sh\necho stub\n")

    with patch("any_guardrail.providers.llamafile.urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.side_effect = [
            _stub_response({"status": "ok"}),
            _stub_response({"choices": [{"message": {"content": "x"}}]}),
        ]
        provider = LlamafileProvider(binary_path=str(binary), port=23471)
        provider.load_model("ibm-granite/granite-guardian-4.1-8b")
        provider.generate_chat(
            messages=[{"role": "user", "content": "x"}],
            max_new_tokens=10,
            do_sample=True,
        )

    body = json.loads(mock_urlopen.call_args_list[1].args[0].data)
    # No temperature in body means the llamafile server uses its own default.
    assert "temperature" not in body
    provider.close()


def test_generate_chat_pins_temperature_zero_in_greedy_mode(tmp_path: Any, fake_subprocess: Any) -> None:
    """Greedy decoding (do_sample=False, the default) must pin temperature=0."""
    binary = tmp_path / "fake.llamafile"
    binary.write_bytes(b"#!/bin/sh\necho stub\n")

    with patch("any_guardrail.providers.llamafile.urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.side_effect = [
            _stub_response({"status": "ok"}),
            _stub_response({"choices": [{"message": {"content": "x"}}]}),
        ]
        provider = LlamafileProvider(binary_path=str(binary), port=23472)
        provider.load_model("ibm-granite/granite-guardian-4.1-8b")
        provider.generate_chat(
            messages=[{"role": "user", "content": "x"}],
            max_new_tokens=10,
        )

    body = json.loads(mock_urlopen.call_args_list[1].args[0].data)
    assert body["temperature"] == 0
    provider.close()


def test_generate_chat_handles_missing_usage_block(tmp_path: Any, fake_subprocess: Any) -> None:
    binary = tmp_path / "fake.llamafile"
    binary.write_bytes(b"#!/bin/sh\necho stub\n")

    with patch("any_guardrail.providers.llamafile.urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.side_effect = [
            _stub_response({"status": "ok"}),
            _stub_response({"choices": [{"message": {"content": "x"}}]}),
        ]
        provider = LlamafileProvider(binary_path=str(binary), port=23468)
        provider.load_model("ibm-granite/granite-guardian-4.1-8b")
        result = provider.generate_chat(
            messages=[{"role": "user", "content": "x"}],
            max_new_tokens=10,
        )

    assert result.data["prompt_token_count"] is None
    assert result.data["completion_token_count"] is None
    provider.close()


def test_generate_chat_before_load_model_raises() -> None:
    provider = LlamafileProvider(binary_path="/fake")
    with pytest.raises(RuntimeError, match="load_model"):
        provider.generate_chat(messages=[{"role": "user", "content": "x"}], max_new_tokens=10)


def test_pre_process_not_supported() -> None:
    provider = LlamafileProvider(binary_path="/fake")
    with pytest.raises(NotImplementedError, match="generate_chat"):
        provider.pre_process("hello")


def test_infer_not_supported() -> None:
    from any_guardrail.types import GuardrailPreprocessOutput

    provider = LlamafileProvider(binary_path="/fake")
    with pytest.raises(NotImplementedError, match="generate_chat"):
        provider.infer(GuardrailPreprocessOutput(data={"messages": []}))


def test_close_terminates_process(tmp_path: Any, fake_subprocess: Any, fake_ready_probe: Any) -> None:
    binary = tmp_path / "fake.llamafile"
    binary.write_bytes(b"#!/bin/sh\necho stub\n")
    _, proc = fake_subprocess

    provider = LlamafileProvider(binary_path=str(binary), port=23469)
    provider.load_model("ibm-granite/granite-guardian-4.1-8b")
    provider.close()

    proc.terminate.assert_called_once()
    assert provider.process is None
    assert provider.base_url is None


def test_close_is_idempotent() -> None:
    provider = LlamafileProvider(binary_path="/fake")
    provider.close()
    provider.close()


def test_close_kills_after_terminate_timeout(tmp_path: Any, fake_ready_probe: Any) -> None:
    """If terminate() doesn't exit cleanly within timeout, kill() is called."""
    binary = tmp_path / "fake.llamafile"
    binary.write_bytes(b"#!/bin/sh\necho stub\n")

    import subprocess as _subprocess

    with patch("any_guardrail.providers.llamafile.subprocess.Popen") as mock_popen:
        proc = MagicMock()
        proc.poll.return_value = None
        proc.wait.side_effect = [_subprocess.TimeoutExpired(cmd="x", timeout=5.0), None]
        mock_popen.return_value = proc

        provider = LlamafileProvider(binary_path=str(binary), port=23470)
        provider.load_model("ibm-granite/granite-guardian-4.1-8b")
        provider.close()

    proc.terminate.assert_called_once()
    proc.kill.assert_called_once()


def test_context_manager_returns_self_from_enter() -> None:
    """`with LlamafileProvider() as p:` should bind `p` to the provider instance."""
    provider = LlamafileProvider(binary_path="/fake")
    with provider as p:
        assert p is provider


def test_context_manager_calls_close_on_exit(tmp_path: Any, fake_subprocess: Any, fake_ready_probe: Any) -> None:
    """Exiting the `with` block should terminate the subprocess."""
    binary = tmp_path / "fake.llamafile"
    binary.write_bytes(b"#!/bin/sh\necho stub\n")
    _, proc = fake_subprocess

    with LlamafileProvider(binary_path=str(binary), port=23473) as provider:
        provider.load_model("ibm-granite/granite-guardian-4.1-8b")
        assert provider.process is not None  # spawned

    proc.terminate.assert_called_once()
    assert provider.process is None
    assert provider.base_url is None


def test_context_manager_calls_close_even_when_block_raises(
    tmp_path: Any, fake_subprocess: Any, fake_ready_probe: Any
) -> None:
    """Exceptions inside the `with` block must still terminate the subprocess."""
    binary = tmp_path / "fake.llamafile"
    binary.write_bytes(b"#!/bin/sh\necho stub\n")
    _, proc = fake_subprocess

    err_msg = "boom"
    provider = LlamafileProvider(binary_path=str(binary), port=23474)

    def _block_that_raises() -> None:
        with provider:
            provider.load_model("ibm-granite/granite-guardian-4.1-8b")
            raise RuntimeError(err_msg)

    with pytest.raises(RuntimeError, match=err_msg):
        _block_that_raises()

    proc.terminate.assert_called_once()
    assert provider.process is None
