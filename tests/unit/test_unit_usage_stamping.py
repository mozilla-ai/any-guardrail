from any_guardrail import GuardrailOutput, GuardrailUsage
from any_guardrail.base import ThreeStageGuardrail
from any_guardrail.types import GuardrailInferenceOutput, GuardrailPreprocessOutput


class FakeGuardrail(ThreeStageGuardrail[dict, dict]):  # type: ignore[type-arg]
    """Minimal three-stage guardrail used to exercise the default pipeline."""

    def __init__(self, usage: GuardrailUsage | None = None) -> None:
        self.model_id = "fake/model"
        self._usage = usage

    def _pre_processing(self, input_text, **kwargs):  # type: ignore[no-untyped-def]
        return GuardrailPreprocessOutput(data={"text": input_text})

    def _inference(self, model_inputs):  # type: ignore[no-untyped-def]
        return GuardrailInferenceOutput(data=model_inputs.data)

    def _post_processing(self, model_outputs):  # type: ignore[no-untyped-def]
        return GuardrailOutput(valid=True, score=0.0, usage=self._usage)


def _single(
    result: GuardrailOutput | list[GuardrailOutput],
) -> GuardrailOutput:
    assert not isinstance(result, list)
    return result


def test_validate_stamps_model_id_and_latency() -> None:
    result = _single(FakeGuardrail().validate("hello"))

    assert result.usage is not None
    assert result.usage.model_id == "fake/model"
    assert result.usage.latency_ms is not None
    assert result.usage.latency_ms >= 0.0


def test_stamping_preserves_guardrail_set_usage_fields() -> None:
    preset = GuardrailUsage(model_id="custom/id", prompt_tokens=11, completion_tokens=3)

    result = _single(FakeGuardrail(usage=preset).validate("hello"))

    assert result.usage is not None
    assert result.usage.model_id == "custom/id"
    assert result.usage.prompt_tokens == 11
    assert result.usage.completion_tokens == 3
    # latency was left None by the guardrail, so the base class fills it.
    assert result.usage.latency_ms is not None


def test_batch_path_stamps_each_item() -> None:
    results = FakeGuardrail().validate(["a", "b"])

    assert isinstance(results, list)
    for result in results:
        assert result.usage is not None
        assert result.usage.model_id == "fake/model"
        assert result.usage.latency_ms is not None


def test_guardrail_without_model_id_stamps_none() -> None:
    guardrail = FakeGuardrail()
    del guardrail.model_id

    result = _single(guardrail.validate("hello"))

    assert result.usage is not None
    assert result.usage.model_id is None
    assert result.usage.latency_ms is not None
