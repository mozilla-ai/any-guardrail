import time
from typing import TYPE_CHECKING, Any, ClassVar

from any_llm import completion
from pydantic import BaseModel, ValidationError

from any_guardrail.base import Guardrail, GuardrailName, GuardrailOutput
from any_guardrail.prompt_registry import PROMPT_REGISTRY, resolve_prompt
from any_guardrail.prompts import PromptSpec
from any_guardrail.registry import GUARDRAIL_METADATA
from any_guardrail.taxonomy import GuardrailMetadata
from any_guardrail.types import GuardrailUsage

if TYPE_CHECKING:
    from any_llm.types.completion import ChatCompletion


# The default system prompt now lives in the import-free prompt registry so it is
# discoverable via ``AnyGuardrail.get_prompt`` and pinnable for benchmarking (#194). This
# alias keeps the public ``DEFAULT_SYSTEM_PROMPT`` name and its text byte-identical.
DEFAULT_SYSTEM_PROMPT = PROMPT_REGISTRY[GuardrailName.ANYLLM].resolve().segments["system"]
"""The any-guardrail policy-judge system prompt (registry default for ``system_prompt``)."""

DEFAULT_MODEL_ID = "openai:gpt-5-nano"
"""Will be used as default argument for `model_id`"""


class GuardrailOutputAnyLLM(BaseModel):
    """Structured-output schema the judge LLM must return (verdict, rationale, and risk score)."""

    valid: bool
    explanation: str
    risk_score: float


class AnyLlm(Guardrail):
    """AnyLlm — policy-based LLM judge that grades text against a natural-language policy using any LLM provider supported by any-llm.

    Wraps ``any_llm.completion`` with structured output: the judge model receives your policy
    inside a system prompt and must return a boolean verdict, an explanation, and a risk score.
    Any provider/model reachable through any-llm can be used (default: ``openai:gpt-5-nano``);
    the chosen model must support structured output (``response_format``), and the provider's
    credentials (e.g. ``OPENAI_API_KEY``) must be configured as any-llm expects.

    Verdict mapping: ``valid`` is the judge's verdict (``True`` = compliant with the policy);
    ``score`` is the judge-reported risk in ``[0, 1]`` (higher = more likely violating the
    policy); ``explanation`` is the judge's rationale; ``raw`` holds the full ``ChatCompletion``.
    When the LLM response cannot be parsed into the expected schema, the output fails closed:
    ``valid=False`` with ``extra={"parse_failure": True}``.

    Expected inputs: a single string plus a natural-language ``policy``, both passed per call to
    ``validate`` — this guardrail is stateless and takes no constructor arguments.

    For more information, see:

    - [any-llm on GitHub](https://github.com/mozilla-ai/any-llm).
    """

    METADATA: ClassVar[GuardrailMetadata] = GUARDRAIL_METADATA[GuardrailName.ANYLLM]

    PROMPT: ClassVar[PromptSpec] = PROMPT_REGISTRY[GuardrailName.ANYLLM]

    def validate(
        self,
        input_text: str,
        policy: str,
        model_id: str = DEFAULT_MODEL_ID,
        system_prompt: str | None = None,
        prompt_version: str | None = None,
        **kwargs: Any,
    ) -> GuardrailOutput:
        """Validate the `input_text` against the given `policy`.

        Args:
            input_text (str): The text to validate (a single string; sent as the user message).
            policy (str): Natural-language policy to validate against, e.g.
                ``"The text must not request or contain personal data."``. Substituted into the
                system prompt's ``{policy}`` placeholder.
            model_id (str, optional): The judge model in any-llm's ``provider:model`` format,
                e.g. ``"openai:gpt-5-nano"`` (default) or ``"mistral:mistral-small-latest"``.
                The model must support structured output.
            system_prompt (str, optional): Override the system prompt. Defaults to ``None``,
                which uses the registry default (:data:`DEFAULT_SYSTEM_PROMPT`) — or the version
                selected by ``prompt_version``. A supplied prompt should have a ``{policy}``
                placeholder and instruct the model to return the ``valid`` / ``explanation`` /
                ``risk_score`` fields of :class:`GuardrailOutputAnyLLM`.
            prompt_version (str, optional): Registered prompt version to use when ``system_prompt``
                is not given. Defaults to ``None`` (the default version). See
                :meth:`any_guardrail.AnyGuardrail.list_prompt_versions`.
            **kwargs: Additional keyword arguments passed through to the `any_llm.completion`
                function (e.g. ``api_key``, ``temperature``).

        Returns:
            GuardrailOutput where ``score`` is the LLM-reported risk in [0, 1]
            (higher = more likely violating the policy). When the LLM response
            cannot be parsed, the output fails closed (``valid=False`` with
            ``extra={"parse_failure": True}``).

        """
        if system_prompt is None:
            system_prompt = resolve_prompt(GuardrailName.ANYLLM, version=prompt_version).segments["system"]
        start = time.perf_counter()
        result: ChatCompletion = completion(  # type: ignore[assignment]
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt.format(policy=policy)},
                {"role": "user", "content": input_text},
            ],
            response_format=GuardrailOutputAnyLLM,
            **kwargs,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0
        usage = GuardrailUsage(model_id=model_id, latency_ms=latency_ms)
        content = result.choices[0].message.content or ""
        try:
            parsed = GuardrailOutputAnyLLM.model_validate_json(content)
        except ValidationError:
            return GuardrailOutput(
                valid=False,
                explanation=content,
                extra={"parse_failure": True},
                usage=usage,
                raw=result,
            )
        return GuardrailOutput(
            valid=parsed.valid,
            explanation=parsed.explanation,
            score=parsed.risk_score,
            usage=usage,
            raw=result,
        )
