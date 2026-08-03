"""Uniform (prompt, response) dispatcher for driving any guardrail's validate() (issue #230).

Judges come in at least six incompatible ``validate()`` shapes: text-pair
(``input_text``/``output_text``), text-pair plus structured kwargs, single-input,
union-typed primary argument, keyed-dict (FlowJudge), and signature-locked pairs
with a different parameter name (OffTopic, LettuceDetect). A caller that wants to
drive judges interchangeably has to hard-code that mapping itself, and the
``**kwargs`` variants mean a typo'd argument is silently ignored rather than
rejected.

``build_validate_call`` centralizes that mapping in one tested, versioned place
instead of leaving every downstream consumer to duplicate (and get wrong) which
argument name is "the response" for each guardrail. It does not remove the
underlying heterogeneity — each guardrail's own ``validate()`` is unchanged —
it only builds the ``(args, kwargs)`` to call it with from a generic
``(prompt, response)`` pair, using the existing ``required_validate_kwargs``
metadata to fail with a clear error instead of a raw ``TypeError`` when something
required is missing.
"""

from collections.abc import Callable
from typing import Any

from any_guardrail.base import Guardrail, GuardrailName
from any_guardrail.registry import GUARDRAIL_METADATA

_Builder = Callable[[GuardrailName, Guardrail, str, str | None, dict[str, Any]], tuple[tuple[Any, ...], dict[str, Any]]]


class EvaluateArgumentError(ValueError):
    """Raised when a call to ``AnyGuardrail.evaluate()`` doesn't supply what the guardrail needs."""


def _pair(response_kwarg: str) -> _Builder:
    """Builder for ``validate(prompt, **{response_kwarg: response}, **kwargs)`` shapes."""

    def build(
        name: GuardrailName, guardrail: Guardrail, prompt: str, response: str | None, kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        del name, guardrail
        call_kwargs = dict(kwargs)
        if response is not None and response_kwarg not in call_kwargs:
            call_kwargs[response_kwarg] = response
        return (prompt,), call_kwargs

    return build


def _single() -> _Builder:
    """Builder for ``validate(prompt, **kwargs)`` shapes with no response/second-text slot."""

    def build(
        name: GuardrailName, guardrail: Guardrail, prompt: str, response: str | None, kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        del guardrail
        if response is not None:
            msg = (
                f"{name.value}.validate() has no response/second-text argument; "
                f"got response={response!r}. Pass extra arguments via evaluate(..., **kwargs) instead."
            )
            raise EvaluateArgumentError(msg)
        return (prompt,), dict(kwargs)

    return build


def _lettuce_detect(
    name: GuardrailName, guardrail: Guardrail, prompt: str, response: str | None, kwargs: dict[str, Any]
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """LettuceDetect's roles are swapped: its primary positional is the answer being checked.

    ``response`` (the answer) fills the primary positional; ``prompt`` (the question the
    answer responds to) fills the ``question`` kwarg; ``context`` (required) must come via
    ``**kwargs``.
    """
    del guardrail
    if response is None:
        msg = (
            f"{name.value}.validate() requires `response` (the answer to check for "
            "hallucinations) via evaluate(..., response=...)."
        )
        raise EvaluateArgumentError(msg)
    call_kwargs = dict(kwargs)
    call_kwargs.setdefault("question", prompt)
    return (response,), call_kwargs


def _flowjudge(
    name: GuardrailName, guardrail: Guardrail, prompt: str, response: str | None, kwargs: dict[str, Any]
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """FlowJudge has no free-text primary argument; ``prompt`` doesn't map onto its keyed dicts.

    Callers must pass ``inputs=[...]`` (a ``list[dict[str, str]]`` matching the guardrail's
    ``required_inputs``) via ``**kwargs`` — checked explicitly here since ``inputs`` is the
    primary positional and isn't tracked by ``required_validate_kwargs``. ``response`` fills
    ``output={guardrail.required_output: response}`` when given and not already supplied.
    """
    del prompt
    if "inputs" not in kwargs:
        msg = (
            f"{name.value}.validate() requires `inputs` (a list[dict[str, str]] matching "
            "required_inputs) via evaluate(..., inputs=[...])."
        )
        raise EvaluateArgumentError(msg)
    call_kwargs = dict(kwargs)
    inputs = call_kwargs.pop("inputs")
    if response is not None and "output" not in call_kwargs:
        call_kwargs["output"] = {getattr(guardrail, "required_output"): response}  # noqa: B009
    return (inputs,), call_kwargs


_BUILDERS: dict[GuardrailName, _Builder] = {
    # Text-pair: validate(input_text, output_text=None, **kwargs) (or no **kwargs).
    GuardrailName.GLIDER: _pair("output_text"),
    GuardrailName.HARMGUARD: _pair("output_text"),
    GuardrailName.DYNA_GUARD: _pair("output_text"),
    GuardrailName.COMPASS_JUDGER: _pair("output_text"),
    GuardrailName.PROMETHEUS: _pair("output_text"),
    GuardrailName.NEMOTRON_CONTENT_SAFETY: _pair("output_text"),
    GuardrailName.KANANA_SAFEGUARD: _pair("output_text"),
    GuardrailName.QWEN3_GUARD: _pair("output_text"),
    GuardrailName.POLY_GUARD: _pair("output_text"),
    GuardrailName.QWEN3_GUARD_STREAM: _pair("output_text"),
    GuardrailName.WILD_GUARD: _pair("output_text"),
    GuardrailName.SELENE: _pair("output_text"),
    GuardrailName.LLAMA_GUARD: _pair("output_text"),
    GuardrailName.GRANITE_GUARDIAN: _pair("output_text"),
    GuardrailName.PATRONUS: _pair("output_text"),
    # Text-pair with a differently-named response kwarg.
    GuardrailName.ALINIA: _pair("output"),
    GuardrailName.OFFTOPIC: _pair("comparison_text"),
    # Single input, no response/second-text slot.
    GuardrailName.BEDROCK_GUARDRAILS: _single(),
    GuardrailName.DEEPSET: _single(),
    GuardrailName.DUOGUARD: _single(),
    GuardrailName.INJECGUARD: _single(),
    GuardrailName.JASPER: _single(),
    GuardrailName.OPENAI_MODERATION: _single(),
    GuardrailName.PANGOLIN: _single(),
    GuardrailName.PROTECTAI: _single(),
    GuardrailName.SENTINEL: _single(),
    GuardrailName.SHIELD_GEMMA: _single(),
    GuardrailName.PROMPT_GUARD: _single(),
    GuardrailName.BIELIK_GUARD: _single(),
    GuardrailName.AZURE_CONTENT_SAFETY: _single(),
    GuardrailName.WATSONX_GUARDIAN: _single(),
    GuardrailName.GLI_GUARD: _single(),
    GuardrailName.GPT_OSS_SAFEGUARD: _single(),
    GuardrailName.GLI_NER_PII: _single(),
    GuardrailName.ANYLLM: _single(),
    GuardrailName.LAKERA_GUARD: _single(),
    GuardrailName.AZURE_PROMPT_SHIELDS: _single(),
    # Bespoke shapes.
    GuardrailName.LETTUCE_DETECT: _lettuce_detect,
    GuardrailName.FLOWJUDGE: _flowjudge,
}


def build_validate_call(
    name: GuardrailName,
    guardrail: Guardrail,
    prompt: str,
    response: str | None,
    kwargs: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Map a generic ``(prompt, response, **kwargs)`` call onto ``guardrail``'s actual ``validate()`` shape.

    Raises:
        EvaluateArgumentError: If ``response`` is supplied for a guardrail with no
            response slot, or if a kwarg the guardrail's ``validate()`` requires
            (per ``GUARDRAIL_METADATA[name].required_validate_kwargs``) is still
            missing after building the call.

    """
    args, call_kwargs = _BUILDERS[name](name, guardrail, prompt, response, kwargs)
    missing = GUARDRAIL_METADATA[name].required_validate_kwargs - call_kwargs.keys()
    if missing:
        msg = f"{name.value}.validate() requires {sorted(missing)}; pass via evaluate(..., response=...) or **kwargs."
        raise EvaluateArgumentError(msg)
    return args, call_kwargs
