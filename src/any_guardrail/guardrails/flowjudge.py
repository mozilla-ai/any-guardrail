from any_guardrail.guardrail import Guardrail
from any_guardrail.types import GuardrailOutput
from flow_judge import FlowJudge, EvalInput
from flow_judge.metrics import Metric, RubricItem  # type: ignore[attr-defined]
from flow_judge.models import Hf
from typing import Dict, List


class FlowJudgeClass(Guardrail):
    """
    Wrapper around FlowJudge, allowing for custom guardrailing based on user defined criteria, metrics, and rubric. Please see
    the model card for more information: [FlowJudge](https://huggingface.co/flowaicom/Flow-Judge-v0.1)

    Args:
        model_id: Name of model. Only used for instantiation of FlowJudge.
        name: User defined metric name.
        criteria: User defined question that they want answered by FlowJudge model.
        rubric: A scoring rubric in a likert scale fashion, providing an integer score and then a description of what the
            value means.
        required_inputs: A list of what is required for the judge to consider.
        required_output: What is the expected output from the judge.

    Raises:
        ValueError: Only supports FlowJudge keywords to instantiate FlowJudge.
    """

    SUPPORTED_MODELS = ["FlowJudge"]

    def __init__(
        self,
        model_id: str,
        name: str,
        criteria: str,
        rubric: Dict[int, str],
        required_inputs: List[str],
        required_output: str,
    ) -> None:
        self.metric_name = name
        self.criteria = criteria
        self.rubric = rubric
        self.required_inputs = required_inputs
        self.required_output = required_output
        self.metric_prompt = self.define_metric_prompt
        # Do super init after setting all attributes, since model_instantiation needs all attributes to be set
        super().__init__(model_id)

    def validate(self, inputs: List[Dict[str, str]], output: Dict[str, str]) -> GuardrailOutput:
        """
        Classifies the desired input and output according to the associated metric provided to the judge.

        Args:
            inputs: A dictionary mapping the required input names to the inputs.
            output: A dictionary mapping the required output name to the output.
        Return:
            A score from the RubricItems and feedback related to the rubric and criteria.
        """
        eval_input = EvalInput(inputs=inputs, output=output)
        result = self.model.evaluate(eval_input, save_results=False)
        return GuardrailOutput(explanation=result.feedback, score=result.score)

    def _load_model(self) -> None:
        """
        Constructs the FlowJudge model using the defined metric prompt that contains the rubric, criteria, and metric.
        Returns:
            judge (FlowJudge): The evaluation model.
        """
        model = Hf(flash_attention=False)
        judge = FlowJudge(metric=self.metric_prompt, model=model)  # type: ignore[arg-type]
        self.model = judge

    def define_metric_prompt(self) -> Metric:
        """
        Constructs the Metric object needed to instantiate the FlowJudge model.
        Returns:
            The Metric object used to construct the FlowJudge model.
        """
        processed_rubric = self._construct_rubric()
        metric_prompt = Metric(
            name=self.metric_name,
            criteria=self.criteria,
            rubric=processed_rubric,
            required_inputs=self.required_inputs,
            required_output=self.required_output,
        )
        return metric_prompt

    def _construct_rubric(self) -> List[RubricItem]:
        """
        Construct the rubric from a user defined rubric dicitionary to construct the Metric object.
        Returns:
            List of RubricItem objects.
        """
        processed_rubric = []
        for key, value in self.rubric.items():
            rubric_item = RubricItem(score=key, description=value)
            processed_rubric.append(rubric_item)
        return processed_rubric
