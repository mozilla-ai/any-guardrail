# Patronus

Patronus — hosted evaluation API running configurable evaluators for hallucination, toxicity, PII, prompt injection, and custom judging (Patronus AI).

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

## Supported Models

- `patronus-evaluate`

## Constructor

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `evaluators` | `list[dict[str, Any]]` | Yes | — | The evaluators to run, each a dict with at least an ``"evaluator"`` key plus optional ``"criteria"`` / ``"explain_strategy"``. Example: ``[{"evaluator": "judge", "criteria": "patronus:prompt-injection"}]``. Must be non-empty. |
| `api_key` | `str | None` | No | `None` | Patronus API key. If ``None``, it is read from the ``PATRONUS_API_KEY`` environment variable. Obtain one at https://app.patronus.ai/ (free Developer tier with starter credit). |
| `endpoint` | `str` | No | `"https://api.patronus.ai/v1/evaluate"` | Evaluate API endpoint URL. Defaults to ``https://api.patronus.ai/v1/evaluate``. |
| `success_strategy` | `Literal['all_pass', 'any_pass']` | No | `"all_pass"` | How to combine multiple evaluators into the overall ``valid`` verdict — ``"all_pass"`` (every evaluator must pass) or ``"any_pass"`` (at least one must). Defaults to ``"all_pass"``. |
| `tags` | `dict[str, str] | None` | No | `None` | Optional tags forwarded with each request for observability, e.g. ``{"env": "prod"}``. |

Initialize the Patronus guardrail.

Does not perform any network I/O — the API is only contacted on
``validate()``.

## validate

Run the configured evaluators against the supplied model interaction.

**Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `input_text` | `str` | Yes | — | The model input (user prompt) to evaluate. |
| `output_text` | `str | None` | No | `None` | The model output to evaluate. Required by evaluators that judge a response (e.g. hallucination, answer relevance). |
| `retrieved_context` | `str | list[str] | None` | No | `None` | RAG context document(s). Required by grounding / hallucination evaluators. |

**Returns:** `GuardrailOutput`
