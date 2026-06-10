from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from any_guardrail.providers.huggingface import HuggingFaceProvider
from any_guardrail.types import GuardrailPreprocessOutput


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


def test_load_model_model_class_override_used_and_self_unchanged(mock_classes: tuple[MagicMock, MagicMock]) -> None:
    """``model_class`` kwarg overrides the provider's default for this call only.

    Lets decoder-LM guardrails (ShieldGemma, GraniteGuardian, LlamaGuard) enforce
    the right loader even when handed a default-constructed (sequence-classification)
    HuggingFaceProvider. The provider's stored ``model_class`` must not be mutated.
    """
    default_model_class, tokenizer_class = mock_classes
    override_model_class = MagicMock()
    override_model_class.from_pretrained.return_value = MagicMock()

    provider = HuggingFaceProvider(model_class=default_model_class, tokenizer_class=tokenizer_class)
    provider.load_model("dummy-id", model_class=override_model_class)

    override_model_class.from_pretrained.assert_called_once()
    default_model_class.from_pretrained.assert_not_called()
    # ``model_class`` kwarg must not leak into from_pretrained as a regular kwarg.
    _, model_kwargs = override_model_class.from_pretrained.call_args
    assert "model_class" not in model_kwargs
    # The provider's default is preserved for subsequent loads.
    assert provider.model_class is default_model_class


def test_load_model_tokenizer_class_override_used_for_this_call_only(
    mock_classes: tuple[MagicMock, MagicMock],
) -> None:
    """``tokenizer_class`` kwarg overrides the provider's default for this call only.

    Needed for Llama Guard 4, which uses ``AutoProcessor`` instead of
    ``AutoTokenizer``. The provider's stored ``tokenizer_class`` must not be mutated.
    """
    model_class, default_tokenizer_class = mock_classes
    override_tokenizer_class = MagicMock()
    override_tokenizer_class.from_pretrained.return_value = MagicMock()

    provider = HuggingFaceProvider(model_class=model_class, tokenizer_class=default_tokenizer_class)
    provider.load_model("dummy-id", tokenizer_class=override_tokenizer_class)

    override_tokenizer_class.from_pretrained.assert_called_once()
    default_tokenizer_class.from_pretrained.assert_not_called()
    _, tokenizer_kwargs = override_tokenizer_class.from_pretrained.call_args
    assert "tokenizer_class" not in tokenizer_kwargs
    assert provider.tokenizer_class is default_tokenizer_class


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


@pytest.mark.parametrize("reserved_key", ["trust_remote_code", "cache_dir", "revision", "torch_dtype"])
def test_reserved_keys_in_model_kwargs_rejected(reserved_key: str) -> None:
    """Reserved keys in ``model_kwargs`` raise ValueError to prevent model/tokenizer mismatches."""
    with pytest.raises(ValueError, match="reserved keys"):
        HuggingFaceProvider(model_kwargs={reserved_key: "anything"})


def test_pre_process_return_tensors_can_be_overridden_via_tokenizer_kwargs() -> None:
    """``return_tensors`` set in ``tokenizer_kwargs`` is honored instead of crashing on a duplicate kwarg."""
    tokenizer = MagicMock(return_value={"input_ids": [[1, 2, 3]]})
    provider = HuggingFaceProvider(tokenizer_kwargs={"return_tensors": "np"})
    provider.tokenizer = tokenizer

    provider.pre_process("hello")

    _, kwargs = tokenizer.call_args
    assert kwargs["return_tensors"] == "np"


def test_pre_process_return_tensors_can_be_overridden_per_call() -> None:
    """``return_tensors`` set as a per-call kwarg is honored instead of crashing on a duplicate kwarg."""
    tokenizer = MagicMock(return_value={"input_ids": [[1, 2, 3]]})
    provider = HuggingFaceProvider()
    provider.tokenizer = tokenizer

    provider.pre_process("hello", return_tensors="np")

    _, kwargs = tokenizer.call_args
    assert kwargs["return_tensors"] == "np"


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


def _provider_with_model_output(logits: torch.Tensor, id2label: dict[int, str] | None = None) -> HuggingFaceProvider:
    """Build a provider whose mocked model returns the given logits tensor."""
    provider = HuggingFaceProvider()
    model = MagicMock()
    model.return_value = SimpleNamespace(logits=logits)
    if id2label is not None:
        model.config = SimpleNamespace(id2label=id2label)
    provider.model = model
    return provider


def test_infer_classification_logits_produce_uniform_shape() -> None:
    """2D logits go through softmax/argmax/label-resolution and produce the uniform shape."""
    logits = torch.tensor([[2.0, 0.5], [0.1, 3.0]])
    provider = _provider_with_model_output(logits, id2label={0: "SAFE", 1: "UNSAFE"})

    result = provider.infer(GuardrailPreprocessOutput(data={"input_ids": torch.tensor([[1, 2, 3]])}))

    assert result.data["logits"].shape == (2, 2)
    assert result.data["scores"].shape == (2, 2)
    assert result.data["predicted_indices"] == [0, 1]
    assert result.data["predicted_labels"] == ["SAFE", "UNSAFE"]
    assert result.data["labels"] == ["SAFE", "UNSAFE"]


def test_infer_causal_lm_3d_logits_skip_label_resolution() -> None:
    """3D logits (causal-LM-shaped) used to crash label-resolution; now they return raw tensor + None fields.

    Regression test for the ShieldGemma integration-test failure: ShieldGemma uses
    AutoModelForCausalLM and reads ``logits[0, -1, [vocab["Yes"], vocab["No"]]]``
    directly, so it needs a torch tensor it can index into.
    """
    # Shape: (batch=1, seq=4, vocab=8) — typical causal-LM forward-pass output.
    logits = torch.randn(1, 4, 8)
    provider = _provider_with_model_output(logits)  # no id2label needed

    result = provider.infer(GuardrailPreprocessOutput(data={"input_ids": torch.tensor([[1, 2, 3, 4]])}))

    # Raw torch tensor so callers can do tensor ops (torch.nn.functional.softmax, slicing).
    assert isinstance(result.data["logits"], torch.Tensor)
    assert result.data["logits"].shape == (1, 4, 8)
    # The classification-only fields don't apply for this shape.
    assert result.data["scores"] is None
    assert result.data["predicted_indices"] is None
    assert result.data["predicted_labels"] is None
    assert result.data["labels"] is None


def test_infer_multi_label_uses_sigmoid_not_softmax() -> None:
    """multi_label=True picks sigmoid so each class probability is independent."""
    logits = torch.tensor([[2.0, 1.0]])
    provider = _provider_with_model_output(logits, id2label={0: "A", 1: "B"})
    provider.multi_label = True

    result = provider.infer(GuardrailPreprocessOutput(data={"input_ids": torch.tensor([[1]])}))

    # sigmoid(2)≈0.881, sigmoid(1)≈0.731 — sum > 1, which softmax can never produce.
    score_sum = float(result.data["scores"][0][0] + result.data["scores"][0][1])
    assert score_sum > 1.4, f"expected sigmoid (sum > 1.4), got sum={score_sum}"
    assert abs(result.data["scores"][0][0] - 0.881) < 0.01
    assert abs(result.data["scores"][0][1] - 0.731) < 0.01


def _make_chat_provider(
    prompt_token_ids: list[int],
    full_output_ids: list[int],
    decoded_text: str = "<score>no</score>",
) -> tuple[HuggingFaceProvider, MagicMock, MagicMock]:
    """Build a HuggingFaceProvider with a tokenizer/model wired up for ``generate_chat`` tests."""
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = {
        "input_ids": torch.tensor([prompt_token_ids]),
        "attention_mask": torch.tensor([[1] * len(prompt_token_ids)]),
    }
    tokenizer.decode.return_value = decoded_text

    model = MagicMock()
    model.generate.return_value = torch.tensor([full_output_ids])

    provider = HuggingFaceProvider()
    provider.tokenizer = tokenizer
    provider.model = model
    return provider, tokenizer, model


def test_generate_chat_decodes_only_new_tokens() -> None:
    """The provider slices off prompt tokens and decodes only the generated tail."""
    provider, tokenizer, model = _make_chat_provider(
        prompt_token_ids=[1, 2, 3, 4],
        full_output_ids=[1, 2, 3, 4, 5, 6, 7],  # 3 newly generated tokens
        decoded_text="<score>no</score>",
    )

    result = provider.generate_chat(
        messages=[{"role": "user", "content": "ping"}],
        max_new_tokens=10,
    )

    # apply_chat_template was given the default template kwargs.
    _, template_kwargs = tokenizer.apply_chat_template.call_args
    assert template_kwargs["add_generation_prompt"] is True
    assert template_kwargs["tokenize"] is True
    assert template_kwargs["return_dict"] is True
    assert template_kwargs["return_tensors"] == "pt"

    # generate was given the full inputs dict plus the explicit generation params.
    _, gen_kwargs = model.generate.call_args
    assert gen_kwargs["max_new_tokens"] == 10
    assert gen_kwargs["do_sample"] is False
    assert "input_ids" in gen_kwargs
    assert "attention_mask" in gen_kwargs

    # decode was passed only the suffix beyond the 4-token prompt.
    decode_args = tokenizer.decode.call_args
    sliced = decode_args.args[0]
    assert sliced.tolist() == [5, 6, 7]
    assert decode_args.kwargs["skip_special_tokens"] is True

    assert result.data["generated_text"] == "<score>no</score>"
    assert result.data["prompt_token_count"] == 4
    assert result.data["completion_token_count"] == 3


def test_generate_chat_forwards_chat_template_kwargs() -> None:
    """Caller-supplied ``chat_template_kwargs`` override the defaults and pass through."""
    provider, tokenizer, _ = _make_chat_provider(
        prompt_token_ids=[1, 2],
        full_output_ids=[1, 2, 3],
    )

    documents = [{"doc_id": "0", "text": "ctx"}]
    provider.generate_chat(
        messages=[{"role": "user", "content": "x"}],
        max_new_tokens=5,
        chat_template_kwargs={"add_generation_prompt": False, "documents": documents},
    )

    _, template_kwargs = tokenizer.apply_chat_template.call_args
    # Override takes effect; documents passed through.
    assert template_kwargs["add_generation_prompt"] is False
    assert template_kwargs["documents"] == documents
    # Other defaults still present.
    assert template_kwargs["return_tensors"] == "pt"
    assert template_kwargs["return_dict"] is True


def test_generate_chat_forwards_generation_kwargs() -> None:
    """``generation_kwargs`` (e.g. pad_token_id) reach ``model.generate``."""
    provider, _, model = _make_chat_provider(
        prompt_token_ids=[1, 2],
        full_output_ids=[1, 2, 3],
    )

    provider.generate_chat(
        messages=[{"role": "user", "content": "x"}],
        max_new_tokens=5,
        generation_kwargs={"pad_token_id": 0},
    )

    _, gen_kwargs = model.generate.call_args
    assert gen_kwargs["pad_token_id"] == 0


def test_generate_chat_forwards_temperature_when_sampling() -> None:
    """``temperature`` is forwarded only when ``do_sample=True``."""
    provider, _, model = _make_chat_provider(
        prompt_token_ids=[1, 2],
        full_output_ids=[1, 2, 3],
    )

    provider.generate_chat(
        messages=[{"role": "user", "content": "x"}],
        max_new_tokens=5,
        do_sample=True,
        temperature=0.7,
    )

    _, gen_kwargs = model.generate.call_args
    assert gen_kwargs["temperature"] == 0.7

    # Greedy path must not forward temperature.
    model.generate.reset_mock()
    provider.generate_chat(
        messages=[{"role": "user", "content": "x"}],
        max_new_tokens=5,
        temperature=0.7,
    )
    _, gen_kwargs_greedy = model.generate.call_args
    assert "temperature" not in gen_kwargs_greedy


def test_generate_chat_returns_raw_output() -> None:
    """``data['raw']`` exposes the unmodified generate() tensor for advanced use cases."""
    provider, _, _ = _make_chat_provider(
        prompt_token_ids=[1, 2],
        full_output_ids=[1, 2, 9, 10],
    )

    result = provider.generate_chat(
        messages=[{"role": "user", "content": "x"}],
        max_new_tokens=5,
    )

    assert result.data["raw"].tolist() == [[1, 2, 9, 10]]
