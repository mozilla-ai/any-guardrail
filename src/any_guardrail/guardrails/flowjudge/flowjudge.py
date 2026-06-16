from typing import TYPE_CHECKING, Any

try:
    from flow_judge import EvalInput, FlowJudge
    from flow_judge.metrics import Metric, RubricItem  # type: ignore[attr-defined]
    from flow_judge.models import Hf

    MISSING_PACKAGES_ERROR = None
except ImportError as e:
    MISSING_PACKAGES_ERROR = e

from any_guardrail.base import GuardrailOutput, ThreeStageGuardrail
from any_guardrail.guardrails.utils import normalize_rubric_to_risk
from any_guardrail.types import GuardrailInferenceOutput, GuardrailPreprocessOutput

if TYPE_CHECKING:
    from flow_judge import EvalInput as EvalInputType
    from flow_judge import EvalOutput as EvalOutputType


class Flowjudge(ThreeStageGuardrail["EvalInputType", "EvalOutputType"]):
    """Wrapper around FlowJudge, allowing for custom guardrailing based on user defined criteria, metrics, and rubric.

    Please see the model card for more information: [FlowJudge](https://huggingface.co/flowaicom/Flow-Judge-v0.1).

    Two ways to specify the evaluation. Either supply the convenience fields
    (``name``/``criteria``/``rubric``/``required_inputs``/``required_output``) to
    build a metric, or pass a prebuilt ``metric`` — a ``flow_judge`` ``Metric`` /
    ``CustomMetric`` or one of the library's preset metrics (e.g.
    ``RESPONSE_FAITHFULNESS_3POINT``).

    Args:
        name: User defined metric name (convenience path).
        criteria: User defined question that they want answered by FlowJudge model (convenience path).
        rubric: A scoring rubric in a likert scale fashion, providing an integer score and then a description of what the
            value means (convenience path).
        required_inputs: A list of what is required for the judge to consider (convenience path).
        required_output: What is the expected output from the judge (convenience path).
        pass_threshold: The rubric score at which the text counts as passing. ``valid`` is
            ``rubric_score >= pass_threshold`` (or ``<=`` when ``higher_is_better`` is False).
        higher_is_better: Whether higher rubric scores mean better/passing text. Set to
            False for rubrics where higher scores mean worse text.
        metric: A prebuilt ``flow_judge`` ``Metric`` / ``CustomMetric`` (e.g. a preset). When
            given, the convenience fields are not required and are ignored.
        model: A prebuilt ``flow_judge`` backend (``Hf``, ``Vllm``, ``Llamafile``, ``Baseten``).
            Defaults to ``Hf(flash_attn=False)``. Use this to pick a faster/quantized backend
            (install the matching ``flow-judge[vllm|llamafile|baseten]`` extra yourself).
        generation_params: Generation parameters (``temperature``, ``top_p``, ``max_new_tokens``,
            ``do_sample``) for the default ``Hf`` backend. Ignored when ``model`` is supplied.

    Raises:
        ImportError: When the ``flowjudge`` extra is not installed.
        ValueError: When neither ``metric`` nor the full convenience field set is provided.

    """

    def __init__(
        self,
        name: str | None = None,
        criteria: str | None = None,
        rubric: dict[int, str] | None = None,
        required_inputs: list[str] | None = None,
        required_output: str | None = None,
        pass_threshold: int = 3,
        higher_is_better: bool = True,
        *,
        metric: Any | None = None,
        model: Any | None = None,
        generation_params: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the FlowJudgeClass."""
        if MISSING_PACKAGES_ERROR is not None:
            msg = "Missing packages for FlowJudge guardrail. You can try `pip install 'any-guardrail[flowjudge]'`"
            raise ImportError(msg) from MISSING_PACKAGES_ERROR

        if metric is not None:
            self.metric_prompt = metric
            # Keep ``self.rubric`` populated so ``_post_processing`` can infer the
            # likert bounds from the metric's rubric items, same as the convenience path.
            self.rubric = {item.score: item.description for item in metric.rubric}
        # Explicit ``is None`` chain (not ``None in (...)``) so the type checker narrows
        # each field to non-None in the ``else`` branch.
        elif name is None or criteria is None or rubric is None or required_inputs is None or required_output is None:
            msg = "Provide either a prebuilt `metric=` or all of name/criteria/rubric/required_inputs/required_output."
            raise ValueError(msg)
        else:
            self.metric_name = name
            self.criteria = criteria
            self.rubric = rubric
            self.required_inputs = required_inputs
            self.required_output = required_output
            self.metric_prompt = self._define_metric_prompt()

        self.pass_threshold = pass_threshold
        self.higher_is_better = higher_is_better
        self.model = self._load_model(model, generation_params)

    def validate(self, inputs: list[dict[str, str]], output: dict[str, str]) -> GuardrailOutput:  # type: ignore[override]
        """Classifies the desired input and output according to the associated metric provided to the judge.

        Args:
            inputs: A dictionary mapping the required input names to the inputs.
            output: A dictionary mapping the required output name to the output.

        Return:
            GuardrailOutput where ``valid`` maps the rubric score through
            ``pass_threshold``, ``score`` is the rubric normalized onto the
            canonical risk axis (using the rubric's integer keys as bounds),
            ``explanation`` is the judge's feedback, and ``extra["rubric_score"]``
            is the raw rubric integer.

        """
        return self._execute(inputs, output)

    def _load_model(self, backend: Any | None, generation_params: dict[str, Any] | None) -> FlowJudge:
        """Construct the FlowJudge model from the metric prompt and a backend.

        Args:
            backend: A prebuilt ``flow_judge`` backend, or None to build a default
                ``Hf(flash_attn=False)`` (``flash_attn=False`` keeps it portable to
                non-Ampere GPUs and CPU).
            generation_params: Generation params for the default ``Hf`` backend; ignored
                when ``backend`` is supplied.

        Returns:
            judge: The evaluation model.

        """
        resolved_backend = backend if backend is not None else Hf(flash_attn=False, generation_params=generation_params)
        return FlowJudge(metric=self.metric_prompt, model=resolved_backend)

    def _define_metric_prompt(self) -> Metric:
        """Construct the Metric object needed to instantiate the FlowJudge model.

        Returns:
            The Metric object used to construct the FlowJudge model.

        """
        processed_rubric = self._construct_rubric()
        return Metric(
            name=self.metric_name,
            criteria=self.criteria,
            rubric=processed_rubric,
            required_inputs=self.required_inputs,
            required_output=self.required_output,
        )

    def _construct_rubric(self) -> list[RubricItem]:
        """Construct the rubric from a user defined rubric dicitionary to construct the Metric object.

        Returns:
            List of RubricItem objects.

        """
        processed_rubric = []
        for key, value in self.rubric.items():
            rubric_item = RubricItem(score=key, description=value)
            processed_rubric.append(rubric_item)
        return processed_rubric

    def _pre_processing(
        self, inputs: list[dict[str, str]], output: dict[str, str]
    ) -> GuardrailPreprocessOutput["EvalInputType"]:
        eval_input = EvalInput(inputs=inputs, output=output)
        return GuardrailPreprocessOutput(data=eval_input)

    def _inference(
        self, eval_input: GuardrailPreprocessOutput["EvalInputType"]
    ) -> GuardrailInferenceOutput["EvalOutputType"]:
        result = self.model.evaluate(eval_input.data, save_results=False)
        return GuardrailInferenceOutput(data=result)

    def _post_processing(self, model_outputs: GuardrailInferenceOutput["EvalOutputType"]) -> GuardrailOutput:
        rubric_score = model_outputs.data.score
        feedback = model_outputs.data.feedback
        if rubric_score is None:
            return GuardrailOutput(valid=False, explanation=feedback, extra={"parse_failure": True})
        passed = rubric_score >= self.pass_threshold if self.higher_is_better else rubric_score <= self.pass_threshold
        # The rubric's integer keys give the scale, so normalize the likert score
        # onto the canonical risk axis (higher = riskier) for a portable `score`.
        score = normalize_rubric_to_risk(
            rubric_score, min(self.rubric), max(self.rubric), higher_is_better=self.higher_is_better
        )
        return GuardrailOutput(valid=passed, score=score, explanation=feedback, extra={"rubric_score": rubric_score})
