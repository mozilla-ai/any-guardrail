import os
from typing import Any, ClassVar, Literal, cast

from any_guardrail.base import Guardrail, GuardrailOutput

CleanlabQualityPreset = Literal["base", "low", "medium", "high", "best"]
CleanlabTask = Literal["default", "classification", "code_generation"]


class CleanlabTlm(Guardrail[bool, dict[str, Any], float]):
    """Wraps Cleanlab's Trustworthy Language Model (TLM) for hallucination / trustworthiness scoring.

    Cleanlab TLM is a hosted scoring service that returns a scalar trustworthiness score in ``[0, 1]``
    for any ``(prompt, response)`` pair. It is designed to detect hallucinations and low-confidence
    answers in RAG and agent pipelines.

    This guardrail is a **hallucination / trustworthiness scoring** guardrail — it is NOT a content
    moderation classifier. It does not detect harmful content, prompt injection, jailbreaks, or
    policy violations. It estimates how confident the model is that a given ``response`` is correct
    for a given ``prompt``.

    Output semantics are **inverted** relative to most safety classifiers in ``any-guardrail``:
    a higher score means MORE trustworthy and therefore MORE valid. ``valid`` is ``True`` when the
    score is greater than or equal to ``threshold``.

    Authentication: requires a Cleanlab TLM API key. Pass via the ``api_key=`` constructor argument
    or set the ``CLEANLAB_TLM_API_KEY`` environment variable. Sign up at https://tlm.cleanlab.ai/
    (Cleanlab offers $5 of free credits at signup; usage is pay-as-you-go thereafter).

    Research backing:
        - Chen & Mueller, *Quantifying Uncertainty in Answers from any Language Model and Enhancing
          their Trustworthiness*, ACL 2024 — the BSDetector methodology underlying TLM.
          https://aclanthology.org/2024.acl-long.283/
        - Cleanlab agent-architecture hallucination benchmarks — TLM detects incorrect RAG responses
          with ~3x the precision of RAGAS / groundedness baselines.
          https://cleanlab.ai/blog/agent-tlm-hallucination-benchmarking/
        - TLM documentation: https://help.cleanlab.ai/tlm/

    Args:
        model_id: One of ``CleanlabTlm.SUPPORTED_MODELS`` (``"TLM"`` or ``"TLM-Lite"``). Defaults to
            ``"TLM"``. ``"TLM-Lite"`` is the cheaper variant.
        api_key: Cleanlab TLM API key. If ``None``, falls back to ``CLEANLAB_TLM_API_KEY`` env var.
        quality_preset: One of ``{"base", "low", "medium", "high", "best"}``. Higher quality presets
            consume more credits per call. Defaults to ``"medium"``.
        task: One of ``{"default", "classification", "code_generation"}``. Tunes TLM for a specific
            downstream task type. Defaults to ``"default"``.

    """

    SUPPORTED_MODELS: ClassVar = ["TLM", "TLM-Lite"]

    def __init__(
        self,
        model_id: str | None = None,
        api_key: str | None = None,
        quality_preset: CleanlabQualityPreset = "medium",
        task: CleanlabTask = "default",
    ) -> None:
        """Initialize the Cleanlab TLM guardrail and instantiate its underlying client.

        Args:
            model_id: One of ``CleanlabTlm.SUPPORTED_MODELS``. Defaults to ``"TLM"``.
            api_key: Cleanlab TLM API key. If ``None``, falls back to the ``CLEANLAB_TLM_API_KEY``
                environment variable. Raises ``ValueError`` if no key is found.
            quality_preset: Quality preset for scoring. One of
                ``{"base", "low", "medium", "high", "best"}``.
            task: Task type for scoring. One of ``{"default", "classification", "code_generation"}``.

        Raises:
            ValueError: If no API key is supplied and ``CLEANLAB_TLM_API_KEY`` is not set.
            ImportError: If the ``cleanlab-tlm`` package is not installed. Install with
                ``pip install 'any-guardrail[cleanlab-tlm]'``.

        """
        if model_id is None:
            self.model_id = self.SUPPORTED_MODELS[0]
        elif model_id not in self.SUPPORTED_MODELS:
            msg = f"Only supports {self.SUPPORTED_MODELS}, got '{model_id}'."
            raise ValueError(msg)
        else:
            self.model_id = model_id

        resolved_key = api_key if api_key is not None else os.getenv("CLEANLAB_TLM_API_KEY")
        if not resolved_key:
            msg = (
                "Cleanlab TLM API key must be provided either via the `api_key=` constructor "
                "argument or the `CLEANLAB_TLM_API_KEY` environment variable. "
                "Sign up at https://tlm.cleanlab.ai/ to obtain a key."
            )
            raise ValueError(msg)
        self.api_key = resolved_key
        self.quality_preset = quality_preset
        self.task = task

        try:
            from cleanlab_tlm import TLM
        except ImportError as exc:
            msg = (
                "The `cleanlab-tlm` package is required to use the CleanlabTlm guardrail. "
                "Install it with `pip install 'any-guardrail[cleanlab-tlm]'`."
            )
            raise ImportError(msg) from exc

        self.tlm = TLM(
            api_key=self.api_key,
            quality_preset=self.quality_preset,
            task=self.task,
        )

    def validate(
        self,
        prompt: str,
        response: str,
        threshold: float = 0.7,
    ) -> GuardrailOutput[bool, dict[str, Any], float]:
        """Score the trustworthiness of ``response`` given ``prompt`` via Cleanlab TLM.

        Note:
            Output semantics are inverted relative to most ``any-guardrail`` guardrails: a HIGHER
            trustworthiness score means MORE trustworthy (and therefore MORE valid). ``valid`` is
            ``True`` when ``score >= threshold``.

        Args:
            prompt: The prompt presented to the LLM.
            response: The response produced by the LLM that we want to score for trustworthiness.
            threshold: Minimum trustworthiness score in ``[0, 1]`` for the response to be considered
                valid. Defaults to ``0.7``.

        Returns:
            A ``GuardrailOutput`` with:

            - ``valid``: ``True`` if ``trustworthiness_score >= threshold``, else ``False``.
            - ``score``: The raw trustworthiness score from TLM.
            - ``explanation``: The ``log`` dict returned by TLM (which may include a free-text
              ``"explanation"`` field) merged with the trustworthiness score for convenience.

        """
        raw_result = self.tlm.get_trustworthiness_score(prompt=prompt, response=response)
        # ``get_trustworthiness_score`` returns ``TLMScore | list[TLMScore]`` depending on whether
        # the inputs were strings or sequences. We always pass single strings here, so the result is
        # the dict-shaped TLMScore branch.
        result = cast("dict[str, Any]", raw_result)
        raw_score = result.get("trustworthiness_score")
        if raw_score is None:
            msg = f"Cleanlab TLM did not return a trustworthiness score. Full response: {result!r}"
            raise RuntimeError(msg)
        score = float(raw_score)
        valid = score >= threshold
        log_dict = cast("dict[str, Any]", result.get("log", {})) or {}
        explanation: dict[str, Any] = {**log_dict, "trustworthiness_score": score}
        return GuardrailOutput(valid=valid, explanation=explanation, score=score)
