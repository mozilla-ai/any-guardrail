try:
    from flow_judge import EvalInput, EvalOutput, FlowJudge
    from flow_judge.metrics import Metric, RubricItem  # type: ignore[attr-defined]
    from flow_judge.models import Hf

    MISSING_PACKAGES_ERROR = None
except ImportError as e:
    MISSING_PACKAGES_ERROR = e

from any_guardrail.base import Guardrail, GuardrailOutput


class Flowjudge(Guardrail):
    """Wrapper around FlowJudge, allowing for custom guardrailing based on user defined criteria, metrics, and rubric.

    Please see the model card for more information: [FlowJudge](https://huggingface.co/flowaicom/Flow-Judge-v0.1).

    Args:
        name: User defined metric name.
        criteria: User defined question that they want answered by FlowJudge model.
        rubric: A scoring rubric in a likert scale fashion, providing an integer score and then a description of what the
            value means.
        required_inputs: A list of what is required for the judge to consider.
        required_output: What is the expected output from the judge.

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
        self.metric_prompt = self._define_metric_prompt()
        self.model = self._load_model()

    def validate(self, inputs: list[dict[str, str]], output: dict[str, str]) -> GuardrailOutput:
        """Classifies the desired input and output according to the associated metric provided to the judge.

        Args:
            inputs: A dictionary mapping the required input names to the inputs.
            output: A dictionary mapping the required output name to the output.

        Return:
            A score from the RubricItems and feedback related to the rubric and criteria.

        """
        eval_input = self._pre_processing(inputs, output)
        result = self._inference(eval_input)
        return GuardrailOutput(explanation=result.feedback, score=result.score)

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

    def _pre_processing(self, inputs: list[dict[str, str]], output: dict[str, str]) -> EvalInput:
        return EvalInput(inputs=inputs, output=output)

    def _inference(self, eval_input: EvalInput) -> EvalOutput:
        return self.model.evaluate(eval_input, save_results=False)
