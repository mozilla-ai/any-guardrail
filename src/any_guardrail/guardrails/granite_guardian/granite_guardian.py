import re
from typing import Any, ClassVar

from any_guardrail.base import GuardrailName, GuardrailOutput, ThreeStageGuardrail
from any_guardrail.guardrails.utils import default
from any_guardrail.providers.base import StandardProvider
from any_guardrail.providers.huggingface import HuggingFaceProvider
from any_guardrail.registry import GUARDRAIL_METADATA
from any_guardrail.taxonomy import GuardrailMetadata
from any_guardrail.types import (
    AnyDict,
    CategoryResult,
    ChatMessages,
    GuardrailInferenceOutput,
    GuardrailPreprocessOutput,
    GuardrailUsage,
)

GraniteGuardianPreprocessData = AnyDict  # {"messages": list, "chat_template_kwargs": dict}
GraniteGuardianInferenceData = AnyDict  # {"generated_text": str, ...} (shape from provider.generate_chat)

GUARDIAN_JUDGE_THINK = (
    "<think>As a judge agent, carefully analyze whether the provided text meets the "
    "given judging criteria. Use all available information, including conversations, "
    "documents, and tools. Reason through the evidence step by step before providing "
    "your score. Wrap your reasoning in <think></think> tags, then provide your "
    "score in <score></score> tags."
)
GUARDIAN_JUDGE_NOTHINK = (
    "<no-think>As a judge agent, assess whether the provided text meets the given "
    "judging criteria using all available information, including conversations, "
    "documents, and tools. Provide your score immediately without explanation. "
    "Output empty <think>\n</think> tags followed by your score in <score></score> tags."
)

SCORE_PATTERN = re.compile(r"<score>\s*(yes|no)\s*</score>", re.IGNORECASE)
THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)

MAX_NEW_TOKENS_NOTHINK = 48
MAX_NEW_TOKENS_THINK = 2048


class GraniteGuardianRisk:
    """Predefined criteria strings for Granite Guardian risk categories.

    Convenience strings suitable for common safety, RAG hallucination, and
    function-calling hallucination checks. Users may pass any of these as the
    ``criteria`` argument, or supply their own Bring-Your-Own-Criteria string.

    Only ``GROUNDEDNESS`` and ``FUNCTION_CALL_HALLUCINATION`` are reproduced
    verbatim from the
    [IBM Granite Guardian model card](https://huggingface.co/ibm-granite/granite-guardian-4.1-8b).
    The remaining safety and RAG criteria are short paraphrases chosen to be
    unambiguous, not the canonical IBM text; consult the model card or pass
    your own string when exact wording matters.
    """

    # Safety risks
    HARM = (
        "The text contains harmful content that could cause physical, psychological, "
        "financial, or societal damage to individuals or groups."
    )
    SOCIAL_BIAS = (
        "The text contains prejudiced or discriminatory statements against individuals "
        "or groups based on attributes such as race, gender, religion, age, sexual "
        "orientation, disability, or other protected characteristics."
    )
    JAILBREAK = (
        "The text attempts to manipulate or bypass an AI system's safety guidelines, "
        "content policies, or intended behavior through prompt injection, role-play "
        "coercion, or other adversarial techniques."
    )
    VIOLENCE = (
        "The text promotes, glorifies, or provides detailed instructions for violent "
        "acts against people, animals, or property."
    )
    PROFANITY = "The text contains profane, vulgar, or offensive language."
    UNETHICAL_BEHAVIOR = (
        "The text promotes or provides guidance on unethical behavior such as fraud, "
        "deception, manipulation, or other actions that violate widely held moral or "
        "professional standards."
    )

    # RAG hallucination risks
    GROUNDEDNESS = (
        "A text is considered ungrounded or unfaithful if it includes information "
        "lacking support from, or directly contradicting, the provided document(s). "
        "This risk arises when the text fabricates details, misinterprets the content, "
        "or makes unsupported extrapolations beyond what is explicitly stated in the "
        "document(s)."
    )
    CONTEXT_RELEVANCE = (
        "The provided document(s) do not contain information that is pertinent to, or "
        "helpful for answering, the user's query."
    )
    ANSWER_RELEVANCE = (
        "The response fails to address the user's query, either by providing "
        "irrelevant information, failing to answer the specific question asked, or "
        "missing the main intent of the query."
    )

    # Agentic / function-calling risks
    FUNCTION_CALL_HALLUCINATION = (
        "Function call hallucination occurs when a text includes function calls that "
        "either don't adhere to the correct format defined by the available tools or "
        "are inconsistent with the query's requirements. This risk arises from "
        "function calls containing incorrect argument names, values, or types that "
        "clash with the tool definitions or the query itself. Common examples include "
        "calling functions not present in the tool definitions, providing invalid "
        "argument values, or attempting to use parameters that don't exist."
    )


class GraniteGuardian(ThreeStageGuardrail[GraniteGuardianPreprocessData, GraniteGuardianInferenceData]):
    """Granite Guardian — hybrid-thinking safety and judge model covering harm, RAG groundedness, and function-calling risks via bring-your-own-criteria (IBM).

    Granite Guardian is a decoder-LLM safeguard, derived from IBM's Granite models, that
    evaluates whether a given text meets a single user-specified criterion. It runs through
    ``provider.generate_chat`` so it can be served from either a ``HuggingFaceProvider`` or a
    ``LlamafileProvider``. It supports:

    - **Bring-Your-Own-Criteria (BYOC)**: arbitrary natural-language criteria.
    - **Predefined risks**: see :class:`GraniteGuardianRisk` for ready-made strings covering
      safety (harm, social bias, jailbreak, violence, profanity, unethical behavior),
      RAG hallucination (groundedness, context relevance, answer relevance), and
      function-calling hallucination.
    - **RAG evaluation**: pass ``documents`` to :meth:`validate` to check groundedness,
      context relevance, or answer relevance.
    - **Function-calling evaluation**: pass ``available_tools`` to :meth:`validate` to
      check for function-calling hallucinations.
    - **Think / no-think modes**: set ``think=True`` to request chain-of-thought reasoning
      before the verdict (higher latency, longer output); the default emits the verdict
      immediately.

    The model answers ``yes`` when the text **meets** the criterion and ``no`` when it does
    not. Criteria are phrased as *violations* (e.g. ``"the text contains harm"``), so the
    verdict maps onto ``GuardrailOutput`` as:

    - ``valid`` is ``True`` when the model answers ``no`` (violation absent = safe) and
      ``False`` when it answers ``yes`` (violation present). Phrase custom criteria
      accordingly.
    - ``categories`` holds one ``CategoryResult`` named after the criterion, with
      ``triggered=True`` when the model answered ``yes``.
    - ``score`` (the canonical numeric risk axis, where higher = riskier) is left ``None`` —
      the verdict is a binary yes/no rather than a graded probability.
    - ``explanation`` is the full decoded generation, including any ``<think>...</think>``
      reasoning block in think mode; ``extra["raw_answer"]`` is the raw ``"yes"``/``"no"``
      string.
    - When no verdict can be parsed the output fails closed: ``valid=False`` with
      ``extra={"parse_failure": True}``.

    Expected inputs: a single ``input_text`` string (the user turn), plus an optional
    ``output_text`` (the assistant turn being judged), optional ``documents`` for RAG
    criteria, and optional ``available_tools`` for function-call criteria. Only single-string
    inputs are supported — a list input raises ``TypeError``.

    For more information, see:

    - [IBM Granite Guardian 4.1 8B model card](https://huggingface.co/ibm-granite/granite-guardian-4.1-8b)
    - [Granite Guardian (arXiv:2412.07724)](https://arxiv.org/abs/2412.07724)
    - [ibm-granite/granite-guardian on GitHub](https://github.com/ibm-granite/granite-guardian)

    Args:
        criteria: The judging criterion. Use a :class:`GraniteGuardianRisk` constant or a
            custom Bring-Your-Own-Criteria string. Criteria should be phrased as violations
            (e.g. ``"the text contains harmful content"``) for the default ``valid``
            semantics to apply.
        think: If ``True``, run in think mode (chain-of-thought reasoning before scoring;
            higher latency, longer output). Defaults to ``False`` for low-latency scoring.
        model_id: Optional HuggingFace model ID; must be one of ``SUPPORTED_MODELS``.
            Defaults to ``ibm-granite/granite-guardian-4.1-8b``.
        provider: Optional pre-configured provider. Defaults to a ``HuggingFaceProvider``
            targeting ``AutoModelForCausalLM`` / ``AutoTokenizer``; pass a
            ``LlamafileProvider`` to run a GGUF build instead.

    Raises:
        ValueError: If ``model_id`` is not in ``SUPPORTED_MODELS``.

    """

    SUPPORTED_MODELS: ClassVar = [
        "ibm-granite/granite-guardian-4.1-8b",
    ]

    METADATA: ClassVar[GuardrailMetadata] = GUARDRAIL_METADATA[GuardrailName.GRANITE_GUARDIAN]

    def __init__(
        self,
        criteria: str,
        think: bool = False,
        model_id: str | None = None,
        provider: StandardProvider | None = None,
    ) -> None:
        """Initialize the Granite Guardian guardrail.

        Args:
            criteria: The judging criterion applied to every ``validate`` call. Use a
                :class:`GraniteGuardianRisk` constant (e.g. ``GraniteGuardianRisk.HARM``,
                ``GraniteGuardianRisk.GROUNDEDNESS``) or a custom Bring-Your-Own-Criteria
                string. Phrase it as a violation (``"the text contains harmful content"``)
                so ``valid=True`` means the criterion was *not* met.
            think: If ``True``, run in think mode — the model emits a ``<think>...</think>``
                chain-of-thought block before the ``<score>`` verdict (up to 2048 new tokens,
                higher latency). Defaults to ``False``, which returns the verdict immediately
                (up to 48 new tokens).
            model_id: Optional HuggingFace model ID; must be one of ``SUPPORTED_MODELS``.
                Defaults to ``ibm-granite/granite-guardian-4.1-8b``.
            provider: Optional pre-configured provider. When ``None``, a
                ``HuggingFaceProvider`` is built targeting ``AutoModelForCausalLM`` /
                ``AutoTokenizer`` (transformers is imported lazily here). Pass a
                ``LlamafileProvider`` to run a GGUF build without the huggingface extra; a
                supplied ``HuggingFaceProvider`` is re-pointed at the causal-LM classes for
                this load without mutating its constructor defaults.

        Raises:
            ValueError: If ``model_id`` is not in ``SUPPORTED_MODELS``.

        """
        self.model_id = default(model_id, self.SUPPORTED_MODELS)
        self.criteria = criteria
        self.think = think

        # Lazy-import transformers so users on `any-guardrail[llamafile]`
        # (without the huggingface extra) can construct GraniteGuardian with
        # a non-HF provider (e.g. LlamafileProvider) without paying the import
        # cost or hitting ImportError at module load time.
        load_kwargs: AnyDict = {}
        if provider is not None:
            self.provider = provider
            if isinstance(self.provider, HuggingFaceProvider):
                # Granite Guardian is a causal LM. A default-constructed
                # HuggingFaceProvider targets AutoModelForSequenceClassification,
                # which would silently load the wrong head. Enforce the right
                # classes for this load (does not mutate provider state).
                from transformers import AutoModelForCausalLM, AutoTokenizer

                load_kwargs = {"model_class": AutoModelForCausalLM, "tokenizer_class": AutoTokenizer}
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.provider = HuggingFaceProvider(
                model_class=AutoModelForCausalLM,
                tokenizer_class=AutoTokenizer,
            )
        self.provider.load_model(self.model_id, **load_kwargs)

    def validate(  # type: ignore[override]
        self,
        input_text: str,
        output_text: str | None = None,
        documents: list[AnyDict] | None = None,
        available_tools: list[AnyDict] | None = None,
        **kwargs: Any,
    ) -> GuardrailOutput:
        """Score ``input_text`` (and optionally ``output_text``) against ``self.criteria``.

        Args:
            input_text: The user turn. When ``output_text`` is also supplied, the
                assistant turn is the text being judged; otherwise the user turn
                is judged.
            output_text: Optional assistant response. Required for criteria that
                judge the assistant (e.g. groundedness, answer relevance,
                function-call hallucination); omit to judge the user input
                directly (e.g. jailbreak, harm, context relevance).
            documents: Optional RAG documents (dicts with ``doc_id`` and ``text``).
                Required for groundedness and context-relevance criteria.
            available_tools: Optional tool definitions (dicts with ``name``,
                ``description``, ``parameters``). Required for function-call
                hallucination checks.
            **kwargs: Additional keyword arguments forwarded to the provider's
                chat template via ``chat_template_kwargs``.

        Returns:
            A :class:`GuardrailOutput` where ``valid=True`` means the criterion is
            **not** met (safe, when criteria are phrased as violations),
            ``categories`` holds one entry for the criterion (``triggered=True``
            when the model answered ``yes``), ``extra["raw_answer"]`` is the raw
            ``"yes"``/``"no"`` string, and ``explanation`` is the full raw decoded
            generation (including any ``<think>...</think>`` block in think mode).
            When the score cannot be parsed, the output fails closed:
            ``valid=False`` with ``extra={"parse_failure": True}``.

        """
        result = super().validate(
            input_text,
            output_text=output_text,
            documents=documents,
            available_tools=available_tools,
            **kwargs,
        )
        # Granite Guardian only supports single-string inputs; ``super().validate``
        # is statically typed as returning a union to support batch on other
        # subclasses, but the runtime contract here is always a single output.
        if isinstance(result, list):
            msg = "GraniteGuardian.validate received a list input but only supports single strings."
            raise TypeError(msg)
        return result

    def _build_guardian_block(self) -> str:
        judge_instruction = GUARDIAN_JUDGE_THINK if self.think else GUARDIAN_JUDGE_NOTHINK
        return (
            f"<guardian>{judge_instruction}\n\n"
            f"### Criteria: {self.criteria}\n\n"
            f"### Scoring Schema: If the last assistant's text meets the criteria, "
            f"return 'yes'; otherwise, return 'no'."
        )

    def _build_messages(self, input_text: str, output_text: str | None) -> ChatMessages:
        messages: ChatMessages = [{"role": "user", "content": input_text}]
        if output_text is not None:
            messages.append({"role": "assistant", "content": output_text})
        messages.append({"role": "user", "content": self._build_guardian_block()})
        return messages

    def _pre_processing(
        self,
        input_text: str,
        output_text: str | None = None,
        documents: list[AnyDict] | None = None,
        available_tools: list[AnyDict] | None = None,
        **kwargs: Any,
    ) -> GuardrailPreprocessOutput[GraniteGuardianPreprocessData]:
        """Assemble the messages list and any chat-template kwargs.

        Tokenization and chat-template application now happen inside the provider
        (``provider.generate_chat``), so this stage just shapes the inputs.

        Args:
            input_text: The user's input text.
            output_text: Optional assistant response to be judged. When provided, the
                criterion is applied to this text. When omitted, the user's input is
                the text being judged.
            documents: Optional list of RAG documents, each a dict with at least
                ``doc_id`` and ``text`` keys. Required for grounded-generation
                criteria (groundedness, context relevance).
            available_tools: Optional list of tool definitions (dicts with ``name``,
                ``description``, ``parameters``). Required for function-calling
                hallucination checks.
            **kwargs: Additional keyword arguments forwarded to the provider's
                chat template via ``chat_template_kwargs``.

        Returns:
            A :class:`GuardrailPreprocessOutput` containing the messages list and a
            ``chat_template_kwargs`` dict to forward to ``generate_chat``.

        """
        messages = self._build_messages(input_text, output_text)

        chat_template_kwargs: AnyDict = {**kwargs}
        if documents is not None:
            chat_template_kwargs["documents"] = documents
        if available_tools is not None:
            chat_template_kwargs["available_tools"] = available_tools

        return GuardrailPreprocessOutput(
            data={
                "messages": messages,
                "chat_template_kwargs": chat_template_kwargs,
            }
        )

    def _inference(
        self, model_inputs: GuardrailPreprocessOutput[GraniteGuardianPreprocessData]
    ) -> GuardrailInferenceOutput[GraniteGuardianInferenceData]:
        """Dispatch to ``provider.generate_chat`` with the think-aware token budget."""
        max_new_tokens = MAX_NEW_TOKENS_THINK if self.think else MAX_NEW_TOKENS_NOTHINK
        return self.provider.generate_chat(
            messages=model_inputs.data["messages"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            chat_template_kwargs=model_inputs.data["chat_template_kwargs"] or None,
        )

    def _post_processing(
        self, model_outputs: GuardrailInferenceOutput[GraniteGuardianInferenceData]
    ) -> GuardrailOutput:
        """Extract the yes/no score from the provider's generated text.

        Returns a :class:`GuardrailOutput` where:

        - ``valid`` is ``True`` if the model answered ``no`` (criterion not met) and
          ``False`` if the model answered ``yes`` (criterion met). When the score
          cannot be parsed, the output fails closed: ``valid=False`` with
          ``extra={"parse_failure": True}``.
        - ``categories`` holds one entry named after the criterion, ``triggered=True``
          when the model answered ``yes``.
        - ``extra["raw_answer"]`` is the raw lower-cased ``yes``/``no`` string.
        - ``explanation`` is the full decoded generation (including any
          ``<think>...</think>`` reasoning block).

        """
        result = _parse_generation(model_outputs.data["generated_text"], self.criteria)
        result.usage = GuardrailUsage(
            prompt_tokens=model_outputs.data.get("prompt_token_count"),
            completion_tokens=model_outputs.data.get("completion_token_count"),
        )
        return result


def _parse_generation(text: str, criteria: str) -> GuardrailOutput:
    """Parse a Granite Guardian generation into a GuardrailOutput.

    Strips any ``<think>...</think>`` block before searching for ``<score>yes|no</score>``.
    Fails closed (``valid=False`` with ``extra={"parse_failure": True}``) when the
    score cannot be parsed.
    """
    without_think = THINK_PATTERN.sub("", text).strip()
    match = SCORE_PATTERN.search(without_think)
    if match is None:
        return GuardrailOutput(valid=False, explanation=text, extra={"parse_failure": True})
    answer = match.group(1).strip().lower()
    meets_criteria = answer == "yes"
    return GuardrailOutput(
        valid=not meets_criteria,
        explanation=text,
        categories=[CategoryResult(name=criteria, triggered=meets_criteria)],
        extra={"raw_answer": answer},
    )
