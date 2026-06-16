import time
from typing import TYPE_CHECKING, Any

from any_llm import completion
from pydantic import BaseModel, ValidationError

from any_guardrail.base import Guardrail, GuardrailOutput
from any_guardrail.types import GuardrailUsage

if TYPE_CHECKING:
    from any_llm.types.completion import ChatCompletion


DEFAULT_SYSTEM_PROMPT = """
You are a guardrail designed to ensure that the input text adheres to a specific policy.
Your only task is to validate the input_text, don't try to answer the user query.

Here is the policy: {policy}

You must return the following:

- valid: bool
    If the input text provided by the user doesn't adhere to the policy, you must reject it (mark it as valid=False).

- explanation: str
    A clear explanation of why the input text was rejected or not.

- risk_score: float (0-1)
    How likely the input text is to violate the policy: 0.0 means clearly compliant,
    1.0 means clearly violating.
"""
"""Will be used as default argument for `system_prompt`"""

DEFAULT_MODEL_ID = "openai:gpt-5-nano"
"""Will be used as default argument for `model_id`"""


class GuardrailOutputAnyLLM(BaseModel):
    """Output model for AnyLlm guardrail."""

    valid: bool
    explanation: str
    risk_score: float


class AnyLlm(Guardrail):
    """A guardrail using `any-llm`."""

    def validate(
        self,
        input_text: str,
        policy: str,
        model_id: str = DEFAULT_MODEL_ID,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        **kwargs: Any,
    ) -> GuardrailOutput:
        """Validate the `input_text` against the given `policy`.

        Args:
            input_text (str): The text to validate.
            policy (str): The policy to validate against.
            model_id (str, optional): The model ID to use.
            system_prompt (str, optional): The system prompt to use.
                Expected to have a `{policy}` placeholder.
            **kwargs: Additional keyword arguments to pass to `any_llm.completion` function.

        Returns:
            GuardrailOutput where ``score`` is the LLM-reported risk in [0, 1]
            (higher = more likely violating the policy). When the LLM response
            cannot be parsed, the output fails closed (``valid=False`` with
            ``extra={"parse_failure": True}``).

        """
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
