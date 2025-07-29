from any_guardrail.guardrails.guardrail import Guardrail
from flow_judge import FlowJudge, EvalInput, EvalOutput
from flow_judge.metrics import Metric, RubricItem  # type: ignore[attr-defined]
from flow_judge.models import Hf
from typing import Dict, List


class FlowJudge(Guardrail):
    """
    Wrapper around FlowJudge, allowing for custom guardrailing based on user defined criteria, metrics, and rubric. Please see
    the model card for more information: https://huggingface.co/flowaicom/Flow-Judge-v0.1
    Args:
        modelpath (str): Name of model. Only used for instantiation of FlowJudge.
        name (str): User defined metric name.
        criteria (str): User defined question that they want answered by FlowJudge model.
        rubric (dict): A scoring rubric in a likert scale fashion, providing an integer score and then a description of what the
            value means.
        required_inputs (list): A list of what is required for the judge to consider.
        required_output (str): What is the expected output from the judge.
    """

    def __init__(
        self,
        modelpath: str,
        name: str,
        criteria: str,
        rubric: Dict[int, str],
        required_inputs: List[str],
        required_output: str,
    ) -> None:
        super().__init__(modelpath)
        self.metric_name = name
        self.criteria = criteria
        self.rubric = rubric
        self.required_inputs = required_inputs
        self.required_output = required_output
        self.metric_prompt = self.define_metric_prompt
        if modelpath in ["FlowJudge", "Flowjudge", "flowjudge"]:
            self.model = self._model_instantiation()
        else:
            raise ValueError("You must use one of the following key word arguments: FlowJudge, Flowjudge, flowjudge.")

    def classify(self, input: List[Dict[str, str]], output: Dict[str, str]) -> EvalOutput:
        """
        Classifies the desired input and output according to the associated metric provided to the judge.

        Args:
            input: A dictionary mapping the required input names to the inputs.
            output: A dictionary mapping the required output name to the output.
        Return:
            A FlowJudge output object containing a score and feedback.
        """
        eval_input = EvalInput(inputs=input, output=output)
        result = self.model.evaluate(eval_input, save_results=False)
        return result

    def _model_instantiation(self) -> FlowJudge:
        """
        Constructs the FlowJudge model using the defined metric prompt that contains the rubric, criteria, and metric.
        Returns:
            judge (FlowJudge): The evaluation model.
        """
        model = Hf(flash_attention=False)
        judge = FlowJudge(self.metric_prompt, model)  # type: ignore[arg-type]
        return judge

    def define_metric_prompt(self) -> Metric:
        """
        Constructs the Metric object needed to instantiate the FlowJudge model.
        Returns:
            The Metric object used to construct the FlowJudge model.
        """
        processed_rubric = self._construct_rubric()
        metric_prompt = Metric(
            name=self.name,  # type: ignore[attr-defined]
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
            list: List of RubricItem objects.
        """
        processed_rubric = []
        for key, value in self.rubric.items():
            rubric_item = RubricItem(score=key, description=value)
            processed_rubric.append(rubric_item)
        return processed_rubric
