from typing import Any, ClassVar

from any_guardrail.base import GuardrailName, StandardGuardrail
from any_guardrail.guardrails.utils import default
from any_guardrail.providers.base import StandardProvider
from any_guardrail.providers.huggingface import HuggingFaceProvider
from any_guardrail.registry import GUARDRAIL_METADATA
from any_guardrail.taxonomy import GuardrailMetadata
from any_guardrail.types import (
    AnyDict,
    CategoryResult,
    GuardrailInferenceOutput,
    GuardrailOutput,
    GuardrailPreprocessOutput,
    GuardrailUsage,
    StandardInferenceOutput,
    StandardPreprocessOutput,
)

SUSFACTOR_DEFAULT_THRESHOLD = 0.5
SUSFACTOR_ENCODER_SUBFOLDER = "encoder"
SUSFACTOR_HEAD_FILENAME = "head.pt"

EMBEDDING_DIM = 1024
NUM_CLASSES = 2
DEFAULT_HIDDEN_DIM = 256

# 512 is the encoder's max sequence length; 2 tokens are reserved for [CLS]/[SEP].
MAX_CONTENT_TOKENS = 510
CHUNK_OVERLAP = 50
CHUNK_STRIDE = MAX_CONTENT_TOKENS - CHUNK_OVERLAP


def _chunk_token_ids(token_ids: list[int]) -> list[list[int]]:
    """Split content token ids (no special tokens) into overlapping windows.

    Each window holds at most ``MAX_CONTENT_TOKENS`` ids so that, once [CLS]/[SEP]
    are added back, the chunk fits the encoder's 512-token limit. Windows overlap
    by ``CHUNK_OVERLAP`` ids (stride ``CHUNK_STRIDE``) so a suspicious span that
    straddles a chunk boundary still lands fully inside at least one chunk.
    Sequences that already fit are returned as a single chunk, unchanged.
    """
    if len(token_ids) <= MAX_CONTENT_TOKENS:
        return [token_ids]
    chunks: list[list[int]] = []
    start = 0
    total = len(token_ids)
    while start < total:
        end = min(start + MAX_CONTENT_TOKENS, total)
        chunks.append(token_ids[start:end])
        if end == total:
            break
        start += CHUNK_STRIDE
    return chunks


def _mean_pool(last_hidden_state: Any, attention_mask: Any) -> Any:
    """Mean-pool encoder token embeddings over non-padding positions.

    ``sum(masked token embeddings) / sum(mask)`` — no max-pooling, no L2
    normalization.
    """
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


class Susfactor(StandardGuardrail):
    """Binary prompt-injection and jailbreak classifier using a chunked e5-large encoder with a trained MLP head.

    SusFactor is 0DIN's proprietary classifier: an e5-large ``AutoModel`` encoder feeds a small
    trained MLP head (``1024 -> 256 -> 2`` with GELU) applied to the mean-pooled ``last_hidden_state``.
    Unlike most encoder classifiers here, the two stages are not fused into a single HuggingFace
    sequence-classification checkpoint — the encoder is loaded from an ``encoder/`` subfolder of the
    model repo, and the head's weights (``head.pt``) are downloaded and loaded separately.

    The input text is tokenized untruncated. Sequences that exceed ``MAX_CONTENT_TOKENS`` (510,
    leaving room for [CLS]/[SEP] in the 512-token encoder limit) are split into overlapping chunks
    (chunk size 510, stride 460, 50-token overlap) so no content is silently truncated. Each chunk is
    scored independently; a suspicious verdict on *any* chunk flags the whole input.

    Verdict mapping onto ``GuardrailOutput``:

    - ``score`` (canonical risk: higher = riskier) is the maximum per-chunk ``P(suspicious)`` across
      all chunks.
    - ``valid`` is ``True`` when every chunk's suspicious probability stays below ``threshold``
      (default 0.5).
    - ``categories`` carries a single ``suspicious`` entry with that max score and the overall
      ``triggered`` flag.

    Expected input: a single ``input_text`` string. There is no prompt+response or chat-message mode.

    The model repository (``0dinai/susfactor-e5-large``) is gated on HuggingFace; loading it requires
    an authenticated Hub token available via the standard ``transformers``/``huggingface_hub``
    resolution (environment variable or cached login).

    Args:
        model_id: Optional HuggingFace model ID; must be one of ``SUPPORTED_MODELS``. Defaults to
            ``0dinai/susfactor-e5-large``.
        threshold: Per-chunk suspicious-probability cutoff at or above which a chunk (and therefore
            the whole input) is flagged. Defaults to 0.5.
        provider: Optional pre-configured provider. Defaults to a ``HuggingFaceProvider`` loading the
            ``encoder/`` subfolder of ``model_id`` as a plain ``AutoModel``.

    """

    SUPPORTED_MODELS: ClassVar = ["0dinai/susfactor-e5-large"]

    METADATA: ClassVar[GuardrailMetadata] = GUARDRAIL_METADATA[GuardrailName.SUSFACTOR]

    def __init__(
        self,
        model_id: str | None = None,
        threshold: float = SUSFACTOR_DEFAULT_THRESHOLD,
        provider: StandardProvider | None = None,
    ) -> None:
        """Initialize the Susfactor guardrail.

        Args:
            model_id: Optional HuggingFace model ID. Must be one of ``SUPPORTED_MODELS``; defaults
                to ``0dinai/susfactor-e5-large``.
            threshold: Per-chunk suspicious-probability cutoff at or above which a chunk (and
                therefore the whole input) is flagged unsafe. Defaults to 0.5.
            provider: Optional pre-configured provider. If ``None``, a default ``HuggingFaceProvider``
                is built targeting a plain ``AutoModel`` encoder and the ``encoder/`` subfolder of
                ``model_id`` is loaded. A supplied ``HuggingFaceProvider`` is corrected to
                ``AutoModel``/``AutoTokenizer`` at load time. Regardless of provider type,
                ``provider.tokenizer`` is always reloaded from ``model_id``'s ``encoder/`` subfolder
                after ``load_model()`` runs, to work around ``HuggingFaceProvider.load_model()`` not
                forwarding ``subfolder`` to the tokenizer's ``from_pretrained`` call. The
                classification head (``head.pt``) is always downloaded and loaded separately,
                regardless of which provider is used, since it isn't part of the encoder checkpoint.

        Raises:
            ValueError: If ``model_id`` is not in ``SUPPORTED_MODELS``.

        """
        self.model_id = default(model_id, self.SUPPORTED_MODELS)
        self.threshold = threshold

        # Lazy-import torch/transformers/huggingface_hub so importing Susfactor
        # does not require the huggingface extra at module load time.
        import torch
        from huggingface_hub import hf_hub_download
        from torch import nn
        from transformers import AutoModel, AutoTokenizer

        load_kwargs: AnyDict = {}
        if provider is not None:
            self.provider = provider
            if isinstance(self.provider, HuggingFaceProvider):
                load_kwargs = {"model_class": AutoModel, "tokenizer_class": AutoTokenizer}
        else:
            self.provider = HuggingFaceProvider(model_class=AutoModel, tokenizer_class=AutoTokenizer)
        self.provider.load_model(self.model_id, subfolder=SUSFACTOR_ENCODER_SUBFOLDER, **load_kwargs)

        # HuggingFaceProvider.load_model() does not forward `subfolder` to the
        # tokenizer's from_pretrained call, so the tokenizer above was loaded
        # from the (wrong) repo root. Reload it explicitly from the encoder
        # subfolder.
        self.provider.tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[attr-defined]
            self.model_id, subfolder=SUSFACTOR_ENCODER_SUBFOLDER
        )

        self._head = nn.Sequential(
            nn.Dropout(0.0),
            nn.Linear(EMBEDDING_DIM, DEFAULT_HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(DEFAULT_HIDDEN_DIM, NUM_CLASSES),
        )
        head_path = hf_hub_download(repo_id=self.model_id, filename=SUSFACTOR_HEAD_FILENAME)
        state_dict = torch.load(head_path, map_location="cpu")
        state_dict = {k.removeprefix("classifier."): v for k, v in state_dict.items()}
        self._head.load_state_dict(state_dict)
        self._head.eval()

    def _pre_processing(self, input_text: str) -> StandardPreprocessOutput:
        """Tokenize the full input untruncated and split it into overlapping chunks.

        Args:
            input_text: The text to classify.

        Returns:
            GuardrailPreprocessOutput wrapping ``{"chunks": [...]}``, one
            ``{"input_ids", "attention_mask"}`` tensor pair per chunk, each with
            [CLS]/[SEP] added back and a batch dimension of 1.

        """
        import torch

        tokenizer = self.provider.tokenizer  # type: ignore[attr-defined]
        content_ids: list[int] = tokenizer(input_text, add_special_tokens=False, truncation=False)["input_ids"]
        cls_id = tokenizer.cls_token_id
        sep_id = tokenizer.sep_token_id

        chunks: list[AnyDict] = []
        for token_chunk in _chunk_token_ids(content_ids):
            ids = [cls_id, *token_chunk, sep_id]
            chunks.append(
                {
                    "input_ids": torch.tensor([ids]),
                    "attention_mask": torch.tensor([[1] * len(ids)]),
                }
            )
        return GuardrailPreprocessOutput(data={"chunks": chunks})

    def _inference(self, model_inputs: StandardPreprocessOutput) -> StandardInferenceOutput:
        """Run each chunk through the encoder + MLP head and softmax to P(suspicious).

        Bypasses ``provider.infer()`` (which assumes a fused sequence-classification
        head): the encoder and the trained MLP head are run explicitly per chunk.
        """
        import torch

        chunk_scores: list[float] = []
        with torch.no_grad():
            for chunk in model_inputs.data["chunks"]:
                encoder_output = self.provider.model(**chunk)  # type: ignore[attr-defined]
                pooled = _mean_pool(encoder_output.last_hidden_state, chunk["attention_mask"])
                logits = self._head(pooled)
                probabilities = torch.softmax(logits, dim=-1)
                chunk_scores.append(probabilities[0, 1].item())
        return GuardrailInferenceOutput(data={"chunk_scores": chunk_scores})

    def _post_processing(self, model_outputs: StandardInferenceOutput) -> GuardrailOutput:
        chunk_scores: list[float] = model_outputs.data["chunk_scores"]
        score = max(chunk_scores)
        is_suspicious = any(chunk_score >= self.threshold for chunk_score in chunk_scores)
        return GuardrailOutput(
            valid=not is_suspicious,
            score=score,
            categories=[CategoryResult(name="suspicious", triggered=is_suspicious, score=score)],
            usage=GuardrailUsage(model_id=self.model_id),
        )
