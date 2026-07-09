from typing import ClassVar

from any_guardrail.base import GuardrailOutput, ThreeStageGuardrail
from any_guardrail.guardrails.utils import default
from any_guardrail.providers.base import StandardProvider
from any_guardrail.providers.huggingface import HuggingFaceProvider
from any_guardrail.types import AnyDict, CategoryResult, StandardInferenceOutput, StandardPreprocessOutput

DUOGUARD_CATEGORIES = [
    "Violent crimes",
    "Non-violent crimes",
    "Sex-related crimes",
    "Child sexual exploitation",
    "Specialized advice",
    "Privacy",
    "Intellectual property",
    "Indiscriminate weapons",
    "Hate",
    "Suicide and self-harm",
    "Sexual content",
    "Jailbreak prompts",
]

DUOGUARD_DEFAULT_THRESHOLD = 0.5  # Taken from the DuoGuard model card.


class DuoGuard(ThreeStageGuardrail[AnyDict, AnyDict]):
    """DuoGuard — multilingual multi-label safety classifier scoring text across 12 harm categories including jailbreak prompts.

    DuoGuard is a compact (0.5B-1.5B) classifier built on Qwen 2.5 and Llama 3.2 backbones,
    trained with a two-player reinforcement-learning framework in which a generator and the
    guardrail co-evolve to synthesize multilingual safety data. It emits an independent
    probability (sigmoid) for each of the 12 categories in ``DUOGUARD_CATEGORIES`` — the
    MLCommons-style hazard taxonomy (violent crimes, non-violent crimes, sex-related crimes,
    child sexual exploitation, specialized advice, privacy, intellectual property,
    indiscriminate weapons, hate, suicide and self-harm, sexual content) plus a
    jailbreak-prompt category.

    The models are fine-tuned primarily for English, French, German, and Spanish, with broader
    coverage inherited from the Qwen 2.5 / Llama 3.2 base models.

    Verdict mapping onto ``GuardrailOutput``:

    - ``categories`` carries all 12 categories, each with its sigmoid probability and a
      ``triggered`` flag (probability strictly above ``threshold``).
    - ``valid`` is ``True`` only when no category is triggered.
    - ``score`` (canonical risk: higher = riskier) is the maximum category probability.

    Expected inputs: a single text string, or a ``list[str]`` for batched classification (the
    inherited ``ThreeStageGuardrail.validate`` handles list input). It screens a single body
    of text; it does not take a separate prompt / response pair.

    For more information, see:

    - [DuoGuard model collection](https://huggingface.co/collections/DuoGuard/duoguard-models-67a29ad8bd579a404e504d21).
    - [DuoGuard-0.5B model card](https://huggingface.co/DuoGuard/DuoGuard-0.5B) (default).
    - [DuoGuard-1B-Llama-3.2-transfer model card](https://huggingface.co/DuoGuard/DuoGuard-1B-Llama-3.2-transfer).
    - [DuoGuard-1.5B-transfer model card](https://huggingface.co/DuoGuard/DuoGuard-1.5B-transfer).
    - [DuoGuard: A Two-Player RL-Driven Framework for Multilingual LLM Guardrails (arXiv:2502.05163)](https://arxiv.org/abs/2502.05163).

    Args:
        model_id: Optional HuggingFace model ID from ``SUPPORTED_MODELS``. Defaults to
            ``DuoGuard/DuoGuard-0.5B``.
        threshold: Per-category probability strictly above which a category is flagged.
            Defaults to 0.5 (from the model card).
        provider: Optional pre-configured provider. Defaults to a ``HuggingFaceProvider`` with
            ``multi_label=True`` and the base model's tokenizer.

    """

    SUPPORTED_MODELS: ClassVar = [
        "DuoGuard/DuoGuard-0.5B",
        "DuoGuard/DuoGuard-1B-Llama-3.2-transfer",
        "DuoGuard/DuoGuard-1.5B-transfer",
    ]

    MODELS_TO_TOKENIZER: ClassVar = {
        "DuoGuard/DuoGuard-0.5B": "Qwen/Qwen2.5-0.5B",
        "DuoGuard/DuoGuard-1B-Llama-3.2-transfer": "meta-llama/Llama-3.2-1B",
        "DuoGuard/DuoGuard-1.5B-transfer": "Qwen/Qwen2.5-1.5B",
    }

    def __init__(
        self,
        model_id: str | None = None,
        threshold: float = DUOGUARD_DEFAULT_THRESHOLD,
        provider: StandardProvider | None = None,
    ) -> None:
        """Initialize the DuoGuard guardrail.

        Args:
            model_id: Optional HuggingFace model ID. Must be one of ``SUPPORTED_MODELS``;
                defaults to ``DuoGuard/DuoGuard-0.5B``. The larger
                ``DuoGuard/DuoGuard-1B-Llama-3.2-transfer`` and
                ``DuoGuard/DuoGuard-1.5B-transfer`` variants trade latency for accuracy.
            threshold: Per-category probability strictly above which that category is flagged
                (and the text becomes invalid). Defaults to 0.5, from the model card.
            provider: Optional pre-configured provider. If ``None``, a default
                ``HuggingFaceProvider`` is built with ``multi_label=True`` and the matching
                base tokenizer (see ``MODELS_TO_TOKENIZER``), then the model is loaded eagerly.
                A pad token is set from the EOS token because the Qwen-family tokenizers ship
                without one.

        Raises:
            ValueError: If ``model_id`` is not in ``SUPPORTED_MODELS``.

        """
        self.model_id = default(model_id, self.SUPPORTED_MODELS)
        self.threshold = threshold
        self.provider = provider or HuggingFaceProvider(
            tokenizer_id=self.MODELS_TO_TOKENIZER[self.model_id],
            multi_label=True,
        )
        self.provider.load_model(self.model_id)
        # Qwen-family tokenizers ship without a pad token; HF needs one for padded batching.
        # Encoderfile bakes tokenizer config into the binary, so there's no tokenizer to touch.
        tokenizer = getattr(self.provider, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.pad_token = tokenizer.eos_token

    def _pre_processing(self, input_text: str) -> StandardPreprocessOutput:
        return self.provider.pre_process(input_text)

    def _inference(self, model_inputs: StandardPreprocessOutput) -> StandardInferenceOutput:
        return self.provider.infer(model_inputs)

    def _post_processing(self, model_outputs: StandardInferenceOutput) -> GuardrailOutput:
        # Providers expose per-category probabilities in ``scores``:
        # HuggingFaceProvider applies sigmoid when ``multi_label=True``; the
        # encoderfile binary applies sigmoid internally for multi-label heads.
        probabilities = [float(probability) for probability in model_outputs.data["scores"][0]]
        categories = [
            CategoryResult(name=category, score=probability, triggered=probability > self.threshold)
            for category, probability in zip(DUOGUARD_CATEGORIES, probabilities, strict=True)
        ]
        return GuardrailOutput(
            valid=not any(category.triggered for category in categories),
            score=max(probabilities),
            categories=categories,
        )
