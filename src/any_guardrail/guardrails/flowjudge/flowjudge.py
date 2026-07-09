from typing import TYPE_CHECKING, Any, ClassVar

from any_guardrail.base import GuardrailName
from any_guardrail.registry import GUARDRAIL_METADATA
from any_guardrail.taxonomy import GuardrailMetadata

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
    """Flow Judge — local LLM judge scoring text against user-defined criteria, metrics, and rubrics via the flow-judge library (Flow AI).

    Flow Judge wraps Flow AI's ``flow-judge`` library and its 3.8B evaluator LLM
    (``flowaicom/Flow-Judge-v0.1``, fine-tuned from Phi-3.5-mini). Unlike most guardrails it
    bypasses the ``any-guardrail`` provider layer and drives ``flow_judge`` directly, so the
    ``flowjudge`` extra must be installed (a top-of-module ``ImportError`` is re-raised from
    ``__init__`` with a ``pip install`` hint otherwise).

    Each guardrail is bound to a single ``flow_judge`` ``Metric`` (a criteria string plus a
    Likert ``rubric``). There are two ways to specify it:

    - **Convenience fields**: supply ``name`` / ``criteria`` / ``rubric`` /
      ``required_inputs`` / ``required_output`` and a ``Metric`` is built for you.
    - **Prebuilt metric**: pass ``metric=`` — a ``flow_judge`` ``Metric`` / ``CustomMetric``
      or one of the library's presets (e.g. ``RESPONSE_FAITHFULNESS_3POINT``). The
      convenience fields are then ignored.

    Verdict mapping onto ``GuardrailOutput``:

    - ``valid`` is ``rubric_score >= pass_threshold`` (or ``<=`` when
      ``higher_is_better=False``).
    - ``score`` (canonical risk: higher = riskier) is the Likert score normalized onto
      [0, 1] via ``normalize_rubric_to_risk``, using the rubric's integer keys as the
      scale bounds — inverted when higher rubric values mean better.
    - ``explanation`` is the judge's feedback; ``extra["rubric_score"]`` is the raw integer.
    - When the backend returns no score, the output fails closed: ``valid=False`` with
      ``extra={"parse_failure": True}``.

    Inputs follow the ``flow_judge`` ``EvalInput`` shape rather than a plain string:
    ``validate(inputs, output)`` takes ``inputs`` — a list of single-key dicts, one per
    ``required_inputs`` name (e.g. ``[{"query": "..."}, {"context": "..."}]``) — and
    ``output``, a single-key dict for the ``required_output`` name (e.g.
    ``{"response": "..."}``). The keys must match the metric's declared inputs/output.

    For more information, see:

    - [Flow-Judge-v0.1 model card](https://huggingface.co/flowaicom/Flow-Judge-v0.1)
    - [flowaicom/flow-judge on GitHub](https://github.com/flowaicom/flow-judge)
    - [Flow Judge overview (Flow AI)](https://flow-ai.com/judge)

    Args:
        name: User-defined metric name (convenience path), e.g. ``"faithfulness"``.
        criteria: User-defined question the judge should answer about the data (convenience
            path), e.g. ``"Is the response grounded in the provided context?"``.
        rubric: A Likert-scale scoring rubric as a ``dict[int, str]`` mapping each integer
            score to what it means (convenience path), e.g.
            ``{1: "Not grounded.", 2: "Partly grounded.", 3: "Fully grounded."}``. Its integer
            keys also define the scale bounds used to normalize ``score``.
        required_inputs: The input field names the judge should consider (convenience path),
            e.g. ``["query", "context"]``. Must match the keys passed to ``validate`` as
            ``inputs``.
        required_output: The output field name the judge should grade (convenience path),
            e.g. ``"response"``. Must match the key passed to ``validate`` as ``output``.
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

    METADATA: ClassVar[GuardrailMetadata] = GUARDRAIL_METADATA[GuardrailName.FLOWJUDGE]

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
        """Initialize the Flow Judge guardrail.

        Provide either a prebuilt ``metric=`` or the full set of convenience fields
        (``name`` / ``criteria`` / ``rubric`` / ``required_inputs`` / ``required_output``);
        the metric is built at construction time and the ``flow_judge`` backend is loaded.

        Args:
            name: User-defined metric name (convenience path), e.g. ``"faithfulness"``.
            criteria: User-defined question the judge should answer (convenience path), e.g.
                ``"Is the response grounded in the provided context?"``.
            rubric: A Likert-scale ``dict[int, str]`` mapping each integer score to its
                meaning (convenience path). Its integer keys define the scale bounds used to
                normalize ``score``.
            required_inputs: The input field names the judge should consider (convenience
                path), e.g. ``["query", "context"]``; must match the ``inputs`` keys passed
                to ``validate``.
            required_output: The output field name the judge should grade (convenience path),
                e.g. ``"response"``; must match the ``output`` key passed to ``validate``.
            pass_threshold: The rubric score at which the response counts as passing. ``valid``
                is ``rubric_score >= pass_threshold`` (or ``<=`` when ``higher_is_better`` is
                ``False``). Defaults to ``3``.
            higher_is_better: Whether higher rubric scores mean better responses. Set ``False``
                for rubrics where a higher number is worse. Defaults to ``True``.
            metric: A prebuilt ``flow_judge`` ``Metric`` / ``CustomMetric`` or preset (e.g.
                ``RESPONSE_FAITHFULNESS_3POINT``). When given, the convenience fields are
                ignored. Keyword-only.
            model: A prebuilt ``flow_judge`` backend (``Hf``, ``Vllm``, ``Llamafile``,
                ``Baseten``). Defaults to ``Hf(flash_attn=False)``. Install the matching
                ``flow-judge[vllm|llamafile|baseten]`` extra yourself for non-default
                backends. Keyword-only.
            generation_params: Generation parameters (``temperature``, ``top_p``,
                ``max_new_tokens``, ``do_sample``) for the default ``Hf`` backend; ignored
                when ``model`` is supplied. Keyword-only.

        Raises:
            ImportError: When the ``flowjudge`` extra is not installed.
            ValueError: When neither ``metric`` nor the full convenience field set is provided.

        """
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
            inputs: A list of single-key dictionaries, one per ``required_inputs`` name, each
                mapping that input name to its value, e.g.
                ``[{"query": "What is the capital of France?"}, {"context": "France's capital is Paris."}]``.
            output: A single-key dictionary mapping the ``required_output`` name to the text
                being graded, e.g. ``{"response": "The capital of France is Paris."}``.

        Returns:
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
        """Construct the rubric from a user-defined rubric dictionary to construct the Metric object.

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
        """Wrap the inputs and output in a ``flow_judge`` ``EvalInput``.

        Args:
            inputs: A list of single-key dicts, one per ``required_inputs`` name, e.g.
                ``[{"query": "..."}, {"context": "..."}]``.
            output: A single-key dict mapping the ``required_output`` name to the graded
                text, e.g. ``{"response": "..."}``.

        Returns:
            GuardrailPreprocessOutput wrapping the ``EvalInput`` passed to the judge.

        """
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
