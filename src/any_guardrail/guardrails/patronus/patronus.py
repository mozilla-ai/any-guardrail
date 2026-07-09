import os
import time
from typing import ClassVar, Literal

import requests

from any_guardrail.base import Guardrail, GuardrailName, GuardrailOutput
from any_guardrail.registry import GUARDRAIL_METADATA
from any_guardrail.taxonomy import GuardrailMetadata
from any_guardrail.types import AnyDict, CategoryResult


class Patronus(Guardrail):
    """Patronus — hosted evaluation API running configurable evaluators for hallucination, toxicity, PII, prompt injection, and custom judging (Patronus AI).

    This is the hosted, pay-per-use counterpart to the locally-run
    :class:`~any_guardrail.guardrails.glider.glider.Glider` (GLIDER) judge and the
    Patronus Lynx hallucination model: the same paper-backed evaluators, served as
    managed configurations behind a single ``/v1/evaluate`` endpoint.

    A single request runs one or more *evaluators*. Each evaluator is selected by
    name (e.g. ``"lynx"`` for hallucination, ``"judge"`` for the managed
    LLM-as-a-judge, ``"answer-relevance"``, toxicity / PII evaluators) and an
    optional managed ``criteria`` alias (e.g. ``"patronus:hallucination"``,
    ``"patronus:prompt-injection"``). Each returns a pass/fail verdict, a raw
    score in ``[0, 1]`` (higher is better; below ``0.5`` fails by default), and —
    when ``explain_strategy`` is set — an explanation.

    Auth is via an API key. Obtain one from https://app.patronus.ai/ (free
    Developer tier with starter credit) and set it via ``PATRONUS_API_KEY`` or
    pass it directly.

    ``GuardrailOutput`` mapping:
        - ``valid`` combines the per-evaluator pass flags per ``success_strategy``
          (``"all_pass"`` → every evaluator must pass; ``"any_pass"`` → at least
          one must).
        - ``score`` is the canonical risk of the *riskiest* evaluator,
          ``1 - min(score_raw)`` (since Patronus ``score_raw`` is higher-is-safer).
        - ``categories`` lists one ``CategoryResult`` per evaluator (``name`` =
          its criteria / evaluator id, ``triggered`` = it failed, ``score`` =
          ``1 - score_raw``).
        - ``explanation`` joins the evaluators' explanations when present.
        - ``extra`` carries ``success_strategy`` and a per-evaluator breakdown;
          ``raw`` is the full response body.
        - Fails closed (``valid=False``, ``extra={"parse_failure": True}``) when
          the response has no ``results``.

    Expected input: ``validate`` takes ``input_text`` (the model input / user
    prompt) plus optional ``output_text`` (the model response, required by
    evaluators that judge a response, e.g. hallucination or answer relevance) and
    optional ``retrieved_context`` (RAG document(s) as a str or list[str], required
    by grounding / hallucination evaluators).

    Research backing:
        - Deshpande et al., *GLIDER: Grading LLM Interactions and Decisions using
          Explainable Ranking* (https://arxiv.org/abs/2412.14140, 2024).
        - Ravi et al., *Lynx: An Open Source Hallucination Evaluation Model*
          (https://arxiv.org/abs/2407.08488, 2024).
        - Docs: https://docs.patronus.ai/

    For more information, see:

    - [Patronus platform (API keys, free Developer tier)](https://app.patronus.ai/)
    - [Patronus documentation](https://docs.patronus.ai/)
    - [GLIDER: Grading LLM Interactions and Decisions using Explainable Ranking (arXiv:2412.14140)](https://arxiv.org/abs/2412.14140)
    - [Lynx: An Open Source Hallucination Evaluation Model (arXiv:2407.08488)](https://arxiv.org/abs/2407.08488)

    Args:
        evaluators (list[dict]): The evaluators to run, each a dict with at least
            an ``"evaluator"`` key (plus optional ``"criteria"`` /
            ``"explain_strategy"``). Example:
            ``[{"evaluator": "judge", "criteria": "patronus:prompt-injection"}]``.
        api_key (str | None): Patronus API key. Falls back to ``PATRONUS_API_KEY``.
        endpoint (str): Evaluate API endpoint. Defaults to
            ``https://api.patronus.ai/v1/evaluate``.
        success_strategy ("all_pass" | "any_pass"): How to combine multiple
            evaluators into the ``valid`` verdict. Defaults to ``"all_pass"``.
        tags (dict[str, str] | None): Optional tags forwarded with each request
            for observability.

    """

    SUPPORTED_MODELS: ClassVar = ["patronus-evaluate"]

    METADATA: ClassVar[GuardrailMetadata] = GUARDRAIL_METADATA[GuardrailName.PATRONUS]

    def __init__(
        self,
        evaluators: list[AnyDict],
        api_key: str | None = None,
        endpoint: str = "https://api.patronus.ai/v1/evaluate",
        success_strategy: Literal["all_pass", "any_pass"] = "all_pass",
        tags: dict[str, str] | None = None,
    ) -> None:
        """Initialize the Patronus guardrail.

        Does not perform any network I/O — the API is only contacted on
        ``validate()``.

        Args:
            evaluators: The evaluators to run, each a dict with at least an
                ``"evaluator"`` key plus optional ``"criteria"`` /
                ``"explain_strategy"``. Example:
                ``[{"evaluator": "judge", "criteria": "patronus:prompt-injection"}]``.
                Must be non-empty.
            api_key: Patronus API key. If ``None``, it is read from the
                ``PATRONUS_API_KEY`` environment variable. Obtain one at
                https://app.patronus.ai/ (free Developer tier with starter credit).
            endpoint: Evaluate API endpoint URL. Defaults to
                ``https://api.patronus.ai/v1/evaluate``.
            success_strategy: How to combine multiple evaluators into the overall
                ``valid`` verdict — ``"all_pass"`` (every evaluator must pass) or
                ``"any_pass"`` (at least one must). Defaults to ``"all_pass"``.
            tags: Optional tags forwarded with each request for observability,
                e.g. ``{"env": "prod"}``.

        Raises:
            ValueError: If no API key is provided and ``PATRONUS_API_KEY`` is not
                set, or if ``evaluators`` is empty.

        """
        if api_key:
            self.api_key = api_key
        elif os.getenv("PATRONUS_API_KEY"):
            self.api_key = os.getenv("PATRONUS_API_KEY")  # type: ignore[assignment]
        else:
            msg = (
                "API key must be provided either as the `api_key=` parameter or through the "
                "PATRONUS_API_KEY environment variable. Sign up at https://app.patronus.ai/ to obtain a key."
            )
            raise ValueError(msg)

        if not evaluators:
            msg = "`evaluators` must be a non-empty list of evaluator dicts, e.g. [{'evaluator': 'lynx'}]."
            raise ValueError(msg)

        self.evaluators = evaluators
        self.endpoint = endpoint
        self.success_strategy = success_strategy
        self.tags = tags

    def validate(
        self,
        input_text: str,
        output_text: str | None = None,
        retrieved_context: str | list[str] | None = None,
    ) -> GuardrailOutput:
        """Run the configured evaluators against the supplied model interaction.

        Args:
            input_text (str): The model input (user prompt) to evaluate.
            output_text (str | None): The model output to evaluate. Required by
                evaluators that judge a response (e.g. hallucination, answer
                relevance).
            retrieved_context (str | list[str] | None): RAG context document(s).
                Required by grounding / hallucination evaluators.

        Returns:
            ``GuardrailOutput`` summarizing the evaluators' verdicts (see the
            class docstring for the field mapping).

        """
        start = time.perf_counter()
        params = self._pre_processing(input_text, output_text, retrieved_context)
        response = self._inference(params)
        result = self._post_processing(response)
        self._stamp_usage(result, (time.perf_counter() - start) * 1000.0)
        return result

    def _pre_processing(
        self,
        input_text: str,
        output_text: str | None,
        retrieved_context: str | list[str] | None,
    ) -> AnyDict:
        """Build the JSON payload for the Patronus Evaluate API request.

        Args:
            input_text: The model input (user prompt), mapped to the
                ``evaluated_model_input`` field, e.g. ``"What is the capital of
                France?"``.
            output_text: Optional model response, mapped to
                ``evaluated_model_output``. Include it for evaluators that judge a
                response (hallucination, answer relevance, toxicity of an answer).
            retrieved_context: Optional RAG context — a single string or a list of
                strings — mapped to ``evaluated_model_retrieved_context``. Include
                it for grounding / hallucination evaluators.

        Returns:
            The request payload, including the configured ``evaluators`` and any
            constructor-level ``tags``.

        """
        body: AnyDict = {
            "evaluators": self.evaluators,
            "evaluated_model_input": input_text,
        }
        if output_text is not None:
            body["evaluated_model_output"] = output_text
        if retrieved_context is not None:
            body["evaluated_model_retrieved_context"] = retrieved_context
        if self.tags:
            body["tags"] = self.tags
        return body

    def _inference(self, params: AnyDict) -> requests.Response:
        response = requests.post(
            self.endpoint,
            headers={"X-API-KEY": self.api_key, "accept": "application/json"},
            json=params,
        )
        if response.status_code != 200:
            msg = f"Request to Patronus Evaluate API failed with status code {response.status_code}: {response.text}"
            raise ValueError(msg)
        return response

    def _post_processing(self, response: requests.Response) -> GuardrailOutput:
        body = response.json()
        results = body.get("results") if isinstance(body, dict) else None
        if not results:
            return GuardrailOutput(valid=False, extra={"parse_failure": True}, raw=body)

        categories: list[CategoryResult] = []
        explanations: list[str] = []
        risk_scores: list[float] = []
        passes: list[bool] = []
        breakdown: list[AnyDict] = []

        for result in results:
            if not isinstance(result, dict):
                # Non-dict result entry: malformed -> count as a failed/triggered evaluator.
                passes.append(False)
                categories.append(CategoryResult(name="evaluator", triggered=True, score=None))
                breakdown.append(
                    {"name": "evaluator", "pass": False, "score_raw": None, "explanation": None, "malformed": True}
                )
                continue
            name = result.get("criteria") or result.get("evaluator_id") or result.get("evaluator") or "evaluator"
            evaluation = result.get("evaluation_result")
            if not isinstance(evaluation, dict):
                # Malformed individual result: count it as a failed/triggered
                # evaluator so ``all_pass`` cannot fail open on a partial response.
                passes.append(False)
                categories.append(CategoryResult(name=name, triggered=True, score=None))
                breakdown.append(
                    {"name": name, "pass": False, "score_raw": None, "explanation": None, "malformed": True}
                )
                continue
            passed = bool(evaluation.get("pass"))
            passes.append(passed)

            score_raw = evaluation.get("score_raw")
            risk = (1.0 - float(score_raw)) if isinstance(score_raw, (int, float)) else None
            if risk is not None:
                risk_scores.append(risk)

            explanation = evaluation.get("explanation")
            if isinstance(explanation, str) and explanation:
                explanations.append(explanation)

            categories.append(CategoryResult(name=name, triggered=not passed, score=risk))
            breakdown.append({"name": name, "pass": passed, "score_raw": score_raw, "explanation": explanation})

        if not passes:
            return GuardrailOutput(valid=False, extra={"parse_failure": True}, raw=body)

        valid = all(passes) if self.success_strategy == "all_pass" else any(passes)
        return GuardrailOutput(
            valid=valid,
            score=max(risk_scores) if risk_scores else None,
            explanation="\n\n".join(explanations) if explanations else None,
            categories=categories,
            extra={"success_strategy": self.success_strategy, "breakdown": breakdown},
            raw=body,
        )
