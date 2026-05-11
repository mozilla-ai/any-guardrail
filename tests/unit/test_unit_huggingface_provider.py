from unittest.mock import MagicMock, patch

import pytest

from any_guardrail.providers.huggingface import HuggingFaceProvider


@pytest.fixture
def mock_classes() -> tuple[MagicMock, MagicMock]:
    """Return mocked model and tokenizer classes whose ``from_pretrained`` returns a MagicMock."""
    model_class = MagicMock()
    model_class.from_pretrained.return_value = MagicMock()
    tokenizer_class = MagicMock()
    tokenizer_class.from_pretrained.return_value = MagicMock()
    return model_class, tokenizer_class


def test_defaults_have_no_extra_kwargs(mock_classes: tuple[MagicMock, MagicMock]) -> None:
    """With no surfaced params set, ``from_pretrained`` is called without extras."""
    model_class, tokenizer_class = mock_classes
    provider = HuggingFaceProvider(model_class=model_class, tokenizer_class=tokenizer_class)
    provider.load_model("dummy-id")

    _, model_kwargs = model_class.from_pretrained.call_args
    _, tokenizer_kwargs = tokenizer_class.from_pretrained.call_args
    assert model_kwargs == {}
    assert tokenizer_kwargs == {}


def test_trust_remote_code_is_forwarded(mock_classes: tuple[MagicMock, MagicMock]) -> None:
    """``trust_remote_code`` is passed to both model and tokenizer."""
    model_class, tokenizer_class = mock_classes
    provider = HuggingFaceProvider(model_class=model_class, tokenizer_class=tokenizer_class, trust_remote_code=True)
    provider.load_model("dummy-id")

    _, model_kwargs = model_class.from_pretrained.call_args
    _, tokenizer_kwargs = tokenizer_class.from_pretrained.call_args
    assert model_kwargs["trust_remote_code"] is True
    assert tokenizer_kwargs["trust_remote_code"] is True


def test_torch_dtype_passed_only_to_model(mock_classes: tuple[MagicMock, MagicMock]) -> None:
    """``torch_dtype`` belongs on the model loader, not the tokenizer."""
    model_class, tokenizer_class = mock_classes
    sentinel_dtype = object()
    provider = HuggingFaceProvider(model_class=model_class, tokenizer_class=tokenizer_class, torch_dtype=sentinel_dtype)
    provider.load_model("dummy-id")

    _, model_kwargs = model_class.from_pretrained.call_args
    _, tokenizer_kwargs = tokenizer_class.from_pretrained.call_args
    assert model_kwargs["torch_dtype"] is sentinel_dtype
    assert "torch_dtype" not in tokenizer_kwargs


def test_cache_dir_and_revision_passed_to_both(mock_classes: tuple[MagicMock, MagicMock]) -> None:
    """``cache_dir`` and ``revision`` are forwarded to both model and tokenizer."""
    model_class, tokenizer_class = mock_classes
    provider = HuggingFaceProvider(
        model_class=model_class,
        tokenizer_class=tokenizer_class,
        cache_dir="/some/cache",
        revision="main",
    )
    provider.load_model("dummy-id")

    _, model_kwargs = model_class.from_pretrained.call_args
    _, tokenizer_kwargs = tokenizer_class.from_pretrained.call_args
    assert model_kwargs["cache_dir"] == "/some/cache"
    assert tokenizer_kwargs["cache_dir"] == "/some/cache"
    assert model_kwargs["revision"] == "main"
    assert tokenizer_kwargs["revision"] == "main"


def test_model_kwargs_forwarded_to_model_only(mock_classes: tuple[MagicMock, MagicMock]) -> None:
    """``model_kwargs`` is a model-only forwarding bag for less common params."""
    model_class, tokenizer_class = mock_classes
    provider = HuggingFaceProvider(
        model_class=model_class,
        tokenizer_class=tokenizer_class,
        model_kwargs={"device_map": "auto", "low_cpu_mem_usage": True},
    )
    provider.load_model("dummy-id")

    _, model_kwargs = model_class.from_pretrained.call_args
    _, tokenizer_kwargs = tokenizer_class.from_pretrained.call_args
    assert model_kwargs["device_map"] == "auto"
    assert model_kwargs["low_cpu_mem_usage"] is True
    assert "device_map" not in tokenizer_kwargs


def test_load_model_kwargs_override_provider_defaults(mock_classes: tuple[MagicMock, MagicMock]) -> None:
    """Kwargs passed to ``load_model`` win over surfaced provider defaults on conflict."""
    model_class, tokenizer_class = mock_classes
    provider = HuggingFaceProvider(
        model_class=model_class,
        tokenizer_class=tokenizer_class,
        model_kwargs={"device_map": "auto"},
    )
    provider.load_model("dummy-id", device_map="cuda:0")

    _, model_kwargs = model_class.from_pretrained.call_args
    assert model_kwargs["device_map"] == "cuda:0"


def test_tokenizer_kwargs_applied_per_call() -> None:
    """``tokenizer_kwargs`` are applied to every ``pre_process`` invocation."""
    tokenizer = MagicMock(return_value={"input_ids": [[1, 2, 3]]})
    provider = HuggingFaceProvider(tokenizer_kwargs={"max_length": 128, "truncation": True})
    provider.tokenizer = tokenizer

    provider.pre_process("hello")

    _, kwargs = tokenizer.call_args
    assert kwargs["max_length"] == 128
    assert kwargs["truncation"] is True
    assert kwargs["return_tensors"] == "pt"


def test_per_call_kwargs_override_provider_tokenizer_kwargs() -> None:
    """Per-call kwargs on ``pre_process`` override the provider-level defaults."""
    tokenizer = MagicMock(return_value={"input_ids": [[1, 2, 3]]})
    provider = HuggingFaceProvider(tokenizer_kwargs={"max_length": 128})
    provider.tokenizer = tokenizer

    provider.pre_process("hello", max_length=512)

    _, kwargs = tokenizer.call_args
    assert kwargs["max_length"] == 512


def test_model_kwargs_none_does_not_error() -> None:
    """Passing ``None`` (the default) yields an empty internal dict, not a None reference."""
    provider = HuggingFaceProvider()
    assert provider.model_kwargs == {}
    assert provider.tokenizer_kwargs == {}


def test_provider_does_not_mutate_caller_dicts() -> None:
    """The provider copies caller-supplied dicts so later caller mutations don't leak in."""
    caller_dict = {"device_map": "auto"}
    provider = HuggingFaceProvider(model_kwargs=caller_dict)
    caller_dict["device_map"] = "cpu"

    assert provider.model_kwargs["device_map"] == "auto"


def test_device_defaults_to_none() -> None:
    """When no device is provided, the provider stores ``None`` and does not move anything."""
    provider = HuggingFaceProvider()
    assert provider.device is None


def test_load_model_moves_model_to_device() -> None:
    """When a device is configured, ``load_model`` moves the loaded model to that device."""
    model = MagicMock()
    moved_model = MagicMock()
    model.to.return_value = moved_model
    model_class = MagicMock()
    model_class.from_pretrained.return_value = model
    tokenizer_class = MagicMock()

    provider = HuggingFaceProvider(model_class=model_class, tokenizer_class=tokenizer_class, device="cuda")
    provider.load_model("dummy-model-id")

    model.to.assert_called_once_with("cuda")
    assert provider.model is moved_model


def test_load_model_skips_to_when_no_device() -> None:
    """When no device is configured, ``load_model`` does not call ``.to`` on the model."""
    model = MagicMock()
    model_class = MagicMock()
    model_class.from_pretrained.return_value = model
    tokenizer_class = MagicMock()

    provider = HuggingFaceProvider(model_class=model_class, tokenizer_class=tokenizer_class)
    provider.load_model("dummy-model-id")

    model.to.assert_not_called()


def test_pre_process_moves_tokens_to_device() -> None:
    """Tokenized inputs are moved to the configured device before being returned."""
    tokenized = MagicMock()
    moved_tokens = MagicMock()
    tokenized.to.return_value = moved_tokens
    tokenizer = MagicMock(return_value=tokenized)

    provider = HuggingFaceProvider(device="cpu")
    provider.tokenizer = tokenizer

    result = provider.pre_process("hello")

    tokenized.to.assert_called_once_with("cpu")
    assert result.data is moved_tokens


def test_pre_process_skips_move_when_no_device() -> None:
    """Tokenized inputs are returned untouched when no device is configured."""
    tokenized = MagicMock()
    tokenizer = MagicMock(return_value=tokenized)

    provider = HuggingFaceProvider()
    provider.tokenizer = tokenizer

    result = provider.pre_process("hello")

    tokenized.to.assert_not_called()
    assert result.data is tokenized


@pytest.mark.parametrize("device", ["cpu", "cuda", "mps", "cuda:0"])
def test_device_passed_through(device: str) -> None:
    """The provider stores arbitrary device strings without validation."""
    provider = HuggingFaceProvider(device=device)
    assert provider.device == device


def test_load_model_uses_to_device_with_real_signature() -> None:
    """End-to-end verification using ``patch`` on the real transformers loader interfaces."""
    model = MagicMock()
    moved_model = MagicMock()
    model.to.return_value = moved_model
    with (
        patch(
            "any_guardrail.providers.huggingface.AutoModelForSequenceClassification.from_pretrained",
            return_value=model,
        ),
        patch("any_guardrail.providers.huggingface.AutoTokenizer.from_pretrained", return_value=MagicMock()),
    ):
        provider = HuggingFaceProvider(device="cuda")
        provider.load_model("dummy")

    assert provider.model is moved_model
