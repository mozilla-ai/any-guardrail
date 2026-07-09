from typing import Any, ClassVar

from any_guardrail.base import GuardrailOutput, ThreeStageGuardrail
from any_guardrail.guardrails.utils import default
from any_guardrail.providers.base import StandardProvider
from any_guardrail.providers.huggingface import HuggingFaceProvider
from any_guardrail.types import (
    AnyDict,
    CategoryResult,
    ChatMessages,
    GuardrailInferenceOutput,
    GuardrailPreprocessOutput,
    SpanResult,
)

Qwen3GuardStreamPreprocessData = AnyDict
Qwen3GuardStreamInferenceData = AnyDict

# Canonical risk mapping for the three severity levels.
SEVERITY_RISK = {"Safe": 0.0, "Controversial": 0.5, "Unsafe": 1.0}

# (severity, category, token record) for one moderated response token.
_ResponseVerdict = tuple[str, str | None, AnyDict]


def _last_verdict(result: Any) -> tuple[str | None, str | None]:
    """Extract the newest (severity, category) pair from a ``stream_moderate_from_ids`` result.

    Returns ``(None, None)`` when no risk level is present or it is not one of the
    three known severities. The model's ``"None"`` category (its "no violation"
    marker) is normalized to ``None``.
    """
    if not isinstance(result, dict) or not result.get("risk_level"):
        return None, None
    severity = str(result["risk_level"][-1]).capitalize()
    if severity not in SEVERITY_RISK:
        return None, None
    categories = result.get("category")
    category = categories[-1] if categories else None
    if not isinstance(category, str) or category == "None":
        category = None
    return severity, category


def _to_span(run: AnyDict, output_text: str | None) -> SpanResult:
    return SpanResult(
        start=run["start"],
        end=run["end"],
        text=output_text[run["start"] : run["end"]] if output_text is not None else None,
        label=run["category"],
        score=SEVERITY_RISK[run["severity"]],
    )


def _build_spans(response_verdicts: list[_ResponseVerdict], output_text: str | None) -> list[SpanResult]:
    """Merge consecutive flagged response tokens into character spans over ``output_text``.

    Runs split when the severity or category changes; tokens without offsets
    (template scaffolding, or a tokenizer without offset support) are skipped.
    """
    spans: list[SpanResult] = []
    current: AnyDict | None = None
    for severity, category, record in response_verdicts:
        if severity == "Safe" or record.get("start") is None:
            if current is not None:
                spans.append(_to_span(current, output_text))
                current = None
            continue
        if current is not None and current["severity"] == severity and current["category"] == category:
            current["end"] = record["end"]
        else:
            if current is not None:
                spans.append(_to_span(current, output_text))
            current = {"severity": severity, "category": category, "start": record["start"], "end": record["end"]}
    if current is not None:
        spans.append(_to_span(current, output_text))
    return spans


class Qwen3GuardStream(ThreeStageGuardrail[Qwen3GuardStreamPreprocessData, Qwen3GuardStreamInferenceData]):
    """Qwen3Guard-Stream — token-level streaming safety moderation with span output (Qwen).

    Classifier heads on a Qwen3 backbone (loaded as remote code) that judge the user
    prompt as a whole and every assistant response token individually, each with a
    three-level severity (``Safe`` / ``Controversial`` / ``Unsafe``, where
    ``Controversial`` means harmfulness is context-dependent). The model is multilingual
    (the Qwen3Guard series covers up to 119 languages) and released under Apache-2.0.
    ``validate`` is a non-streaming facade over the token-level streaming API: it feeds
    the whole prompt, then each ``output_text`` token in turn, and aggregates the worst
    severity seen.

    Verdict mapping onto ``GuardrailOutput``:

    - ``valid`` is ``True`` only when everything judged is ``Safe``; when
      ``strict=False``, ``Controversial`` content also passes (only ``Unsafe`` fails).
    - ``score`` maps the worst severity onto the canonical risk axis (higher = riskier):
      ``Safe`` → ``0.0``, ``Controversial`` → ``0.5``, ``Unsafe`` → ``1.0``.
    - ``categories`` lists the distinct violation categories (``triggered=True``) seen
      across the prompt and non-``Safe`` response tokens.
    - ``spans`` (response mode only) merges consecutive flagged response tokens into
      character-offset runs over ``output_text``, each labeled with its category and
      severity-derived ``score``; ``None`` when nothing is flagged or the tokenizer
      cannot supply offsets (a slow tokenizer degrades spans gracefully).
    - ``extra`` carries the worst ``severity``, the ``prompt_severity``, and (in response
      mode) the ``response_severity``.
    - Fails closed (``valid=False`` with ``extra={"parse_failure": True}``) when the
      backend reports no usable risk level.

    For the generative variants (``Qwen3Guard-Gen-*``), see ``Qwen3Guard``.

    Expected inputs: a single ``input_text`` (the user prompt; required) plus an optional
    ``output_text`` (the assistant response). With no ``output_text`` only the prompt is
    moderated and no spans are produced. List/batch input is not supported — passing a
    list raises ``TypeError``.

    HuggingFace-only: the model ships its classification heads as remote code, so a
    user-supplied provider must be a ``HuggingFaceProvider`` constructed with
    ``trust_remote_code=True``. The remote modeling code currently requires
    ``transformers>=4.51,<5`` (transformers 5 removed APIs it relies on); construction
    raises ``ImportError`` on transformers >= 5.

    For more information, see:

    - [Qwen3Guard-Stream-0.6B model card](https://huggingface.co/Qwen/Qwen3Guard-Stream-0.6B) (default).
    - [Qwen3Guard-Stream-4B model card](https://huggingface.co/Qwen/Qwen3Guard-Stream-4B).
    - [Qwen3Guard-Stream-8B model card](https://huggingface.co/Qwen/Qwen3Guard-Stream-8B).
    - [Qwen3Guard Technical Report (arXiv:2510.14276)](https://arxiv.org/abs/2510.14276)
    - [Qwen3Guard: Real-time Safety for Your Token Stream (Qwen blog)](https://qwenlm.github.io/blog/qwen3guard/)

    Args:
        strict: If ``True`` (default), only ``Safe`` verdicts pass validation; set
            ``False`` to let ``Controversial`` content pass (``valid=True``), leaving
            it reflected only in ``score``, ``extra``, and ``spans``.
        model_id: Optional HuggingFace model ID. Must be one of ``SUPPORTED_MODELS``;
            defaults to ``Qwen/Qwen3Guard-Stream-0.6B``.
        provider: Optional pre-configured ``HuggingFaceProvider`` with
            ``trust_remote_code=True``. Defaults to one loading the remote-code model.

    """

    SUPPORTED_MODELS: ClassVar = [
        "Qwen/Qwen3Guard-Stream-0.6B",
        "Qwen/Qwen3Guard-Stream-4B",
        "Qwen/Qwen3Guard-Stream-8B",
    ]

    def __init__(
        self,
        strict: bool = True,
        model_id: str | None = None,
        provider: StandardProvider | None = None,
    ) -> None:
        """Initialize the Qwen3GuardStream guardrail.

        Args:
            strict: If ``True`` (default), only ``Safe`` verdicts pass validation; set
                ``False`` to let ``Controversial`` content pass (``valid=True``), leaving
                it reflected only in ``score``, ``extra``, and ``spans``.
            model_id: Optional HuggingFace model ID. Must be one of ``SUPPORTED_MODELS``;
                defaults to ``Qwen/Qwen3Guard-Stream-0.6B``.
            provider: Optional pre-configured provider. Must be a ``HuggingFaceProvider``
                constructed with ``trust_remote_code=True`` (the classification heads
                load as remote code); it is loaded with ``model_class=AutoModel`` /
                ``tokenizer_class=AutoTokenizer``. Defaults to one loading the
                remote-code model.

        Raises:
            ValueError: If ``model_id`` is not in ``SUPPORTED_MODELS``, or if a supplied
                ``HuggingFaceProvider`` was not constructed with ``trust_remote_code=True``.
            ImportError: If the installed ``transformers`` is version 5 or newer, which
                the remote modeling code cannot run.

        """
        self.model_id = default(model_id, self.SUPPORTED_MODELS)
        self.strict = strict
        self._require_supported_transformers()
        load_kwargs: AnyDict = {}
        if provider is not None:
            if isinstance(provider, HuggingFaceProvider):
                if not provider.trust_remote_code:
                    msg = (
                        "Qwen3Guard-Stream ships its classification heads as remote code; construct the "
                        "provider with HuggingFaceProvider(trust_remote_code=True) so both the model and "
                        "tokenizer load it."
                    )
                    raise ValueError(msg)
                from transformers import AutoModel, AutoTokenizer

                load_kwargs = {"model_class": AutoModel, "tokenizer_class": AutoTokenizer}
            self.provider = provider
        else:
            from transformers import AutoModel, AutoTokenizer

            self.provider = HuggingFaceProvider(
                model_class=AutoModel, tokenizer_class=AutoTokenizer, trust_remote_code=True
            )
        self.provider.load_model(self.model_id, **load_kwargs)

    def validate(  # type: ignore[override]
        self, input_text: str, output_text: str | None = None, **kwargs: Any
    ) -> GuardrailOutput:
        """Moderate a user prompt and, optionally, the assistant response to it.

        Args:
            input_text: The user prompt to moderate, e.g. ``"How do I make a bomb?"``.
                A single string; list/batch input is rejected with ``TypeError``.
            output_text: Optional assistant response moderated token-by-token, e.g.
                ``"I can't help with that."``. When supplied, flagged token runs are
                returned as ``spans`` with character offsets into this text; when omitted,
                only the prompt is moderated and no spans are produced.
            **kwargs: Reserved for forward compatibility; forwarded to the base pipeline
                and otherwise ignored.

        Returns:
            GuardrailOutput where ``valid`` reflects the worst severity under ``strict``,
            ``score`` is the canonical risk for that severity, ``categories`` lists the
            distinct violation categories, ``spans`` marks flagged response runs, and
            ``extra`` carries the per-part severities. Fails closed (``valid=False`` with
            ``extra={"parse_failure": True}``) when no usable risk level is returned.

        Raises:
            TypeError: If a list input is supplied (only single strings are supported).

        """
        result = super().validate(input_text, output_text=output_text, **kwargs)
        if isinstance(result, list):
            msg = "Qwen3GuardStream.validate received a list input but only supports single strings."
            raise TypeError(msg)
        return result

    def _pre_processing(
        self, input_text: str, output_text: str | None = None, **kwargs: Any
    ) -> GuardrailPreprocessOutput[Qwen3GuardStreamPreprocessData]:
        """Build the chat messages for streaming moderation.

        Args:
            input_text: The user prompt, added as the ``user`` turn.
            output_text: Optional assistant response, added as an ``assistant`` turn so
                it can be moderated token-by-token. It is carried forward (with a
                ``has_response`` flag) so response tokens can be mapped back to character
                spans over this text.
            **kwargs: Ignored (accepted for pipeline compatibility).

        Returns:
            GuardrailPreprocessOutput wrapping the chat messages plus the ``has_response``
            flag and the raw ``output_text`` used for span construction.

        """
        del kwargs
        messages: ChatMessages = [{"role": "user", "content": input_text}]
        if output_text is not None:
            messages.append({"role": "assistant", "content": output_text})
        return GuardrailPreprocessOutput(
            data={"messages": messages, "has_response": output_text is not None, "output_text": output_text}
        )

    def _inference(
        self, model_inputs: GuardrailPreprocessOutput[Qwen3GuardStreamPreprocessData]
    ) -> GuardrailInferenceOutput[Qwen3GuardStreamInferenceData]:
        tokenizer = self.provider.tokenizer  # type: ignore[attr-defined]
        model = self.provider.model  # type: ignore[attr-defined]
        has_response: bool = model_inputs.data["has_response"]
        output_text: str | None = model_inputs.data["output_text"]

        text: str = tokenizer.apply_chat_template(
            model_inputs.data["messages"], tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
        try:
            encoding = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
            offsets = encoding["offset_mapping"][0].tolist()
        except NotImplementedError:  # slow tokenizer: spans degrade gracefully
            encoding = tokenizer(text, return_tensors="pt")
            offsets = None
        token_ids = encoding["input_ids"][0]
        device = self.provider.device  # type: ignore[attr-defined]
        if device is not None:
            token_ids = token_ids.to(device)

        user_end_index = self._user_turn_end(tokenizer, token_ids.tolist())
        response_base = text.rfind(output_text) if output_text else -1

        stream_state = None
        response_tokens: list[AnyDict] = []
        try:
            prompt_result, stream_state = model.stream_moderate_from_ids(
                token_ids[: user_end_index + 1], role="user", stream_state=None
            )
            if has_response:
                for index in range(user_end_index + 1, len(token_ids)):
                    result, stream_state = model.stream_moderate_from_ids(
                        token_ids[index], role="assistant", stream_state=stream_state
                    )
                    record: AnyDict = {"result": result, "start": None, "end": None}
                    if offsets is not None and response_base >= 0 and output_text is not None:
                        # Clamp the token's offsets in the templated text onto the
                        # output_text region, shifting to output_text coordinates;
                        # scaffolding tokens outside it keep start=None.
                        start = max(int(offsets[index][0]), response_base)
                        end = min(int(offsets[index][1]), response_base + len(output_text))
                        if start < end:
                            record["start"] = start - response_base
                            record["end"] = end - response_base
                    response_tokens.append(record)
        finally:
            if stream_state is not None:
                model.close_stream(stream_state)
        return GuardrailInferenceOutput(
            data={
                "prompt_result": prompt_result,
                "response_tokens": response_tokens,
                "has_response": has_response,
                "output_text": output_text,
            }
        )

    def _post_processing(
        self, model_outputs: GuardrailInferenceOutput[Qwen3GuardStreamInferenceData]
    ) -> GuardrailOutput:
        data = model_outputs.data
        has_response: bool = data.get("has_response", False)
        output_text: str | None = data.get("output_text")

        prompt_severity, prompt_category = _last_verdict(data.get("prompt_result"))
        if prompt_severity is None:
            return GuardrailOutput(valid=False, extra={"parse_failure": True})
        response_verdicts: list[_ResponseVerdict] = []
        for record in data.get("response_tokens", []):
            severity, category = _last_verdict(record.get("result"))
            if severity is None:
                return GuardrailOutput(valid=False, extra={"parse_failure": True})
            response_verdicts.append((severity, category, record))

        response_severities = [severity for severity, _, _ in response_verdicts]
        worst = max([prompt_severity, *response_severities], key=lambda severity: SEVERITY_RISK[severity])

        names: dict[str, None] = {}
        if prompt_severity != "Safe" and prompt_category is not None:
            names.setdefault(prompt_category)
        for severity, category, _ in response_verdicts:
            if severity != "Safe" and category is not None:
                names.setdefault(category)

        extra: AnyDict = {"severity": worst, "prompt_severity": prompt_severity}
        if has_response:
            extra["response_severity"] = (
                max(response_severities, key=lambda severity: SEVERITY_RISK[severity])
                if response_severities
                else "Safe"
            )
        spans = _build_spans(response_verdicts, output_text) if has_response else []
        return GuardrailOutput(
            valid=worst == "Safe" if self.strict else worst != "Unsafe",
            score=SEVERITY_RISK[worst],
            categories=[CategoryResult(name=name, triggered=True) for name in names],
            spans=spans or None,
            extra=extra,
        )

    def _require_supported_transformers(self) -> None:
        """Raise when the installed transformers cannot run the remote modeling code.

        The Qwen3Guard-Stream model repos target transformers 4.x (>= 4.51 per the
        model card); transformers 5 removed several APIs the remote code relies on
        (the implicit ``pad_token_id`` config default, ``ROPE_INIT_FUNCTIONS["default"]``,
        the old rotary-embedding weight-init interface).
        """
        try:
            from transformers import __version__ as transformers_version
        except ImportError:
            return  # provider construction surfaces the install hint instead
        if int(transformers_version.split(".")[0]) >= 5:
            msg = (
                f"Qwen3Guard-Stream's remote modeling code (in the {self.model_id} model repo, not this "
                f"library) is incompatible with transformers >= 5 (installed: {transformers_version}). "
                "Install 'transformers>=4.51,<5' to use Qwen3GuardStream, or check the model repo for an "
                "updated revision."
            )
            raise ImportError(msg)

    @staticmethod
    def _user_turn_end(tokenizer: Any, token_ids: list[int]) -> int:
        """Index of the ``<|im_end|>`` closing the last user turn (model-card boundary scan)."""
        im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        user_id = tokenizer.convert_tokens_to_ids("user")
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        last_start = next(
            (i for i in range(len(token_ids) - 1, -1, -1) if token_ids[i : i + 2] == [im_start_id, user_id]),
            None,
        )
        if last_start is None:
            msg = "Could not locate the user turn in the tokenized chat template output."
            raise ValueError(msg)
        user_end = next((i for i in range(last_start + 2, len(token_ids)) if token_ids[i] == im_end_id), None)
        if user_end is None:
            msg = "Could not locate the end of the user turn in the tokenized chat template output."
            raise ValueError(msg)
        return user_end
