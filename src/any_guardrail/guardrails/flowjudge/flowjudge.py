from typing import TYPE_CHECKING

try:
    from flow_judge import EvalInput, FlowJudge
    from flow_judge.metrics import Metric, RubricItem  # type: ignore[attr-defined]
    from flow_judge.models import Hf

    MISSING_PACKAGES_ERROR = None
except ImportError as e:
    MISSING_PACKAGES_ERROR = e

from any_guardrail.base import GuardrailOutput, ThreeStageGuardrail
from any_guardrail.types import GuardrailInferenceOutput, GuardrailPreprocessOutput

if TYPE_CHECKING:
    from flow_judge import EvalInput as EvalInputType
    from flow_judge import EvalOutput as EvalOutputType


class Flowjudge(ThreeStageGuardrail["EvalInputType", "EvalOutputType"]):
    """Wrapper around FlowJudge, allowing for custom guardrailing based on user defined criteria, metrics, and rubric.

    Please see the model card for more information: [FlowJudge](https://huggingface.co/flowaicom/Flow-Judge-v0.1).

    Args:
        name: User defined metric name.
        criteria: User defined question that they want answered by FlowJudge model.
        rubric: A scoring rubric in a likert scale fashion, providing an integer score and then a description of what the
            value means.
        required_inputs: A list of what is required for the judge to consider.
        required_output: What is the expected output from the judge.
        pass_threshold: The rubric score at which the text counts as passing. ``valid`` is
            ``rubric_score >= pass_threshold`` (or ``<=`` when ``higher_is_better`` is False).
        higher_is_better: Whether higher rubric scores mean better/passing text. Set to
            False for rubrics where higher scores mean worse text.

    Raises:
        ValueError: Only supports FlowJudge keywords to instantiate FlowJudge.

    """

    def __init__(
        self,
        name: str,
        criteria: str,
        rubric: dict[int, str],
        required_inputs: list[str],
        required_output: str,
        pass_threshold: int,
        higher_is_better: bool = True,
    ) -> None:
        """Initialize the FlowJudgeClass."""
        if MISSING_PACKAGES_ERROR is not None:
            msg = "Missing packages for FlowJudge guardrail. You can try `pip install 'any-guardrail[flowjudge]'`"
            raise ImportError(msg) from MISSING_PACKAGES_ERROR

        self.metric_name = name
        self.criteria = criteria
        self.rubric = rubric
        self.required_inputs = required_inputs
        self.required_output = required_output
        self.pass_threshold = pass_threshold
        self.higher_is_better = higher_is_better
        self.metric_prompt = self._define_metric_prompt()
        self.model = self._load_model()

    def validate(self, inputs: list[dict[str, str]], output: dict[str, str]) -> GuardrailOutput:  # type: ignore[override]
        """Classifies the desired input and output according to the associated metric provided to the judge.

        Args:
            inputs: A dictionary mapping the required input names to the inputs.
            output: A dictionary mapping the required output name to the output.

        Return:
            GuardrailOutput where ``valid`` maps the rubric score through
            ``pass_threshold``, ``explanation`` is the judge's feedback, and
            ``extra["rubric_score"]`` is the raw rubric integer.

        """
        return self._execute(inputs, output)

    def _load_model(self) -> FlowJudge:
        """Construct the FlowJudge model using the defined metric prompt that contains the rubric, criteria, and metric.

        Returns:
            judge: The evaluation model.

        """
        model = Hf(flash_attn=False)
        return FlowJudge(metric=self.metric_prompt, model=model)

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
        return GuardrailOutput(valid=passed, explanation=feedback, extra={"rubric_score": rubric_score})
