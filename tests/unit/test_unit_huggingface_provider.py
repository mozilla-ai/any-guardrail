from unittest.mock import MagicMock, patch

import pytest

from any_guardrail.providers.huggingface import HuggingFaceProvider


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
