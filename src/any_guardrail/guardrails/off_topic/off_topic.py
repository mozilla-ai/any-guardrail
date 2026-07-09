from typing import Any, ClassVar

from any_guardrail.base import GuardrailOutput, ThreeStageGuardrail
from any_guardrail.guardrails.off_topic.off_topic_jina import OffTopicJina
from any_guardrail.guardrails.off_topic.off_topic_stsb import OffTopicStsb
from any_guardrail.guardrails.utils import default
from any_guardrail.providers.base import StandardProvider
from any_guardrail.types import GuardrailInferenceOutput, GuardrailPreprocessOutput


class OffTopic(ThreeStageGuardrail[Any, Any]):
    """Off-Topic — cross-encoder relevance detector that flags whether an input strays from a comparison text (GovTech Singapore).

    A dispatcher over GovTech Singapore's two open off-topic detection models. Given an
    ``input_text`` and a ``comparison_text`` (typically the system prompt or the app's
    intended topic), it decides whether the input is *on-topic* (relevant) or *off-topic*
    (a distraction / topic drift). It selects the implementation from ``model_id``:

    - ``mozilla-ai/jina-embeddings-v2-small-en-off-topic`` → ``OffTopicJina``, a
      bi-encoder that embeds the two texts separately (jina-embeddings-v2-small-en) and
      learns their relationship through cross-attention layers.
    - ``mozilla-ai/stsb-roberta-base-off-topic`` (default) → ``OffTopicStsb``, a
      cross-encoder that concatenates the two texts and scores them jointly with a
      fine-tuned stsb-roberta-base.

    Both are English-language models that truncate long inputs (Jina at 1024 tokens, STSB
    at 514 tokens). Note they emit a ``warnings.warn`` about that truncation limit on every
    call, whether or not the input is actually long enough to be truncated.

    Verdict mapping onto ``GuardrailOutput``:

    - ``valid`` is ``True`` when the input is on-topic, ``False`` when off-topic.
    - ``score`` is ``P(off-topic)`` on the canonical risk axis (higher = riskier, i.e.
      more likely off-topic).
    - ``categories`` reports both class probabilities: ``on-topic`` (score) and
      ``off-topic`` (score, with ``triggered=True`` when off-topic is the argmax class).

    Expected inputs: two single strings. ``comparison_text`` is semantically required
    even though it defaults to ``None`` — ``validate`` raises ``ValueError`` if it is
    missing or empty. List/batch input is not supported.

    For more information, see:

    - [Off-Topic (STSB cross-encoder) model card](https://huggingface.co/mozilla-ai/stsb-roberta-base-off-topic) (default).
    - [Off-Topic (Jina bi-encoder) model card](https://huggingface.co/mozilla-ai/jina-embeddings-v2-small-en-off-topic).
    - [govtech/stsb-roberta-base-off-topic](https://huggingface.co/govtech/stsb-roberta-base-off-topic) (upstream).
    - [govtech/jina-embeddings-v2-small-en-off-topic](https://huggingface.co/govtech/jina-embeddings-v2-small-en-off-topic) (upstream).
    - [A Flexible Large Language Models Guardrail Development Methodology Applied to Off-Topic Prompt Detection (arXiv:2411.12946)](https://arxiv.org/abs/2411.12946)
    - [Open-sourcing an Off-Topic Prompt Guardrail (GovTech blog)](https://medium.com/dsaid-govtech/open-sourcing-an-off-topic-prompt-guardrail-fde422a66152)

    Args:
        model_id: Optional HuggingFace model ID selecting the implementation. Must be one
            of ``SUPPORTED_MODELS``; defaults to ``mozilla-ai/stsb-roberta-base-off-topic``
            (the STSB cross-encoder).
        provider: Reserved for future extensibility; currently unused. Each implementation
            loads its model directly via ``transformers``.

    """

    SUPPORTED_MODELS: ClassVar = [
        "mozilla-ai/jina-embeddings-v2-small-en-off-topic",
        "mozilla-ai/stsb-roberta-base-off-topic",
    ]

    implementation: OffTopicJina | OffTopicStsb

    def __init__(
        self,
        model_id: str | None = None,
        provider: StandardProvider | None = None,  # Reserved for future extensibility
    ) -> None:
        """Initialize the Off-Topic guardrail, selecting the implementation from ``model_id``.

        Args:
            model_id: Optional HuggingFace model ID choosing which detector to load. Must
                be one of ``SUPPORTED_MODELS``:
                ``mozilla-ai/jina-embeddings-v2-small-en-off-topic`` loads the Jina
                bi-encoder (``OffTopicJina``); ``mozilla-ai/stsb-roberta-base-off-topic``
                (the default) loads the STSB cross-encoder (``OffTopicStsb``).
            provider: Reserved for future extensibility; currently unused. The selected
                implementation loads its model directly via ``transformers``.

        Raises:
            ValueError: If ``model_id`` is not in ``SUPPORTED_MODELS``.

        """
        self.model_id = default(model_id, self.SUPPORTED_MODELS)
        self.provider = provider  # Reserved for future extensibility
        if self.model_id == self.SUPPORTED_MODELS[0]:
            self.implementation = OffTopicJina(provider=provider)
        else:
            self.implementation = OffTopicStsb(provider=provider)

    def validate(  # type: ignore[override]
        self, input_text: str, comparison_text: str | None = None
    ) -> GuardrailOutput:
        """Judge whether ``input_text`` is on-topic relative to ``comparison_text``.

        Args:
            input_text: The text to classify, e.g. a user prompt such as
                ``"What's the weather in Paris?"``. A single string.
            comparison_text: The reference topic to compare against — typically the
                system prompt or the app's intended subject, e.g.
                ``"You are a customer-support bot for an online bookstore."``. Although
                it defaults to ``None`` for signature reasons, it is semantically required:
                a missing or empty value raises ``ValueError``.

        Returns:
            GuardrailOutput with ``valid=True`` when on-topic / ``valid=False`` when
            off-topic, ``score`` = ``P(off-topic)`` (higher = riskier), and both class
            probabilities in ``categories``.

        Raises:
            ValueError: If ``comparison_text`` is not provided.

        """
        msg = "Must provide a text to compare to."
        if not comparison_text:
            raise ValueError(msg)
        return self._execute(input_text, comparison_text)

    def _pre_processing(self, *args: Any, **kwargs: Any) -> GuardrailPreprocessOutput[Any]:
        """Delegate pre-processing to the selected implementation."""
        return self.implementation._pre_processing(*args, **kwargs)

    def _inference(self, model_inputs: GuardrailPreprocessOutput[Any]) -> GuardrailInferenceOutput[Any]:
        """Delegate inference to the selected implementation."""
        return self.implementation._inference(model_inputs)

    def _post_processing(self, model_outputs: GuardrailInferenceOutput[Any]) -> GuardrailOutput:
        """Delegate post-processing to the selected implementation."""
        return self.implementation._post_processing(model_outputs)
