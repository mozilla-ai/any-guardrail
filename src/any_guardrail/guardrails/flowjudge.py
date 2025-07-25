"""
Flowjudge guardrail for evaluating outputs using custom metrics and rubrics.
"""
from any_guardrail.guardrails.guardrail import Guardrail
from flow_judge import FlowJudge, EvalInput
from flow_judge.metrics import Metric, RubricItem
from flow_judge.models import Hf
from typing import Any


class Flowjudge(Guardrail):
    """
    Guardrail for evaluating outputs using custom metrics and rubrics.
    Args:
        modelpath (str): Path to the model.
        name (str): Name of the metric.
        criteria (str): Criteria for evaluation.
        rubric (dict): Rubric for scoring.
        required_inputs (list): Required input fields.
        required_outputs (list): Required output fields.
    """
    def __init__(self, modelpath: str, name: str, criteria: str, rubric: dict, required_inputs: list, required_outputs: list) -> None:
        """
        Initialize Flowjudge with model path, metric name, criteria, rubric, and required fields.
        """
        self.modelpath = modelpath
        self.metric_name = name
        self.criteria = criteria
        self.rubric = rubric
        self.required_inputs = required_inputs
        self.required_outputs = required_outputs
        self.metric_prompt = self.define_metric_prompt
        try:
            self.model = self.model_instantiation()
        except Exception as e:
            raise RuntimeError(f"Failed to instantiate model: {e}")

    def classify(self, input: Any, output: Any) -> Any:
        """
        Evaluate the input and output using the loaded model and metric.
        Returns the evaluation result.
        """
        try:
            eval_input = EvalInput(
                inputs = input,
                output=output
            )
            result = self.model.evaluate(eval_input, save_results=False)
            return result
        except Exception as e:
            raise RuntimeError(f"Error during evaluation: {e}")

    def model_instantiation(self) -> Any:
        """
        Instantiate the Flowjudge model with the metric prompt.
        Returns:
            judge (FlowJudge): The evaluation model.
        """
        try:
            model=Hf(flash_attention=False)
            judge = FlowJudge(self.metric_prompt, model)
            return judge
        except Exception as e:
            raise RuntimeError(f"Error instantiating Flowjudge: {e}")

    def define_metric_prompt(self) -> Any:
        """
        Define the metric prompt using the rubric and criteria.
        Returns:
            Metric: The metric prompt object.
        """
        processed_rubric = self._construct_rubric()
        metric_prompt = Metric(
            name=self.name,
            criteria=self.criteria,
            rubric=processed_rubric,
            required_inputs=self.required_inputs,
            required_outputs=self.required_outputs
        )
        return metric_prompt

    def _construct_rubric(self) -> list:
        """
        Construct the rubric from the rubric dictionary.
        Returns:
            list: List of RubricItem objects.
        """
        processed_rubric = []
        for key, value in self.rubric.items():
            rubric_item = RubricItem(
                score = key,
                description=value
            )
            processed_rubric.append(rubric_item)
        return processed_rubric
