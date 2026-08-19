from typing import Any, ClassVar

import numpy as np

from any_guardrail.base import GuardrailName, StandardGuardrail
from any_guardrail.guardrails.utils import default
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


class Susfactor(StandardGuardrail):
    """Binary prompt-injection and jailbreak classifier using a chunked e5-large encoder with a trained MLP head.

    SusFactor is 0DIN's proprietary classifier: an e5-large encoder and its trained MLP head
    (``1024 -> 256 -> 2`` with GELU, applied to the mean-pooled ``last_hidden_state``) are fused
    into a single ONNX graph, exported ahead of time and shipped as ``onnx/model.onnx`` (plus a
    companion ``onnx/model.onnx_data`` external-data file) in the model repository. Unlike the
    ``HuggingFaceProvider``-based guardrails in this codebase, Susfactor bypasses the provider
    layer entirely and runs the fused graph directly through ``onnxruntime.InferenceSession`` — no
    ``torch``/``transformers`` model classes are involved at inference time, only a tokenizer.

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

    See https://0din.ai/docs/defense/introduction for SusFactor's model card, threat-feed
    training methodology, and benchmark results.

    The model repository (``0dinai/susfactor-e5-large-onnx``) is gated on HuggingFace; loading it
    requires an authenticated Hub token available via the standard
    ``transformers``/``huggingface_hub`` resolution (environment variable or cached login).

    Args:
        model_id: Optional HuggingFace model ID; must be one of ``SUPPORTED_MODELS``. Defaults to
            ``0dinai/susfactor-e5-large-onnx``.
        threshold: Per-chunk suspicious-probability cutoff at or above which a chunk (and therefore
            the whole input) is flagged. Defaults to 0.5.
        session: Optional pre-built ``onnxruntime.InferenceSession``. Useful for testing: when both
            ``session`` and ``tokenizer`` are supplied, no download or model loading happens.
        tokenizer: Optional pre-built tokenizer. Useful for testing; see ``session``.

    """

    SUPPORTED_MODELS: ClassVar = ["0dinai/susfactor-e5-large-onnx"]

    METADATA: ClassVar[GuardrailMetadata] = GUARDRAIL_METADATA[GuardrailName.SUSFACTOR]

    def __init__(
        self,
        model_id: str | None = None,
        threshold: float = SUSFACTOR_DEFAULT_THRESHOLD,
        session: Any = None,
        tokenizer: Any = None,
    ) -> None:
        """Initialize the Susfactor guardrail.

        Args:
            model_id: Optional HuggingFace model ID. Must be one of ``SUPPORTED_MODELS``; defaults
                to ``0dinai/susfactor-e5-large-onnx``.
            threshold: Per-chunk suspicious-probability cutoff at or above which a chunk (and
                therefore the whole input) is flagged unsafe. Defaults to 0.5.
            session: Optional pre-built ``onnxruntime.InferenceSession``. If supplied together with
                ``tokenizer``, it is used directly and no download/model loading happens.
            tokenizer: Optional pre-built tokenizer. If supplied together with ``session``, it is
                used directly and no download/model loading happens.

        Raises:
            ValueError: If ``model_id`` is not in ``SUPPORTED_MODELS``.

        """
        self.threshold = threshold

        if session is not None and tokenizer is not None:
            self.model_id = model_id or self.SUPPORTED_MODELS[0]
            self._session = session
            self._tokenizer = tokenizer
            return

        # Lazy-import onnxruntime/huggingface_hub/transformers so importing
        # Susfactor does not require the onnx extra at module load time.
        import onnxruntime
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer

        self.model_id = default(model_id, self.SUPPORTED_MODELS)
        local_dir = snapshot_download(self.model_id)
        self._tokenizer = AutoTokenizer.from_pretrained(local_dir, local_files_only=True)
        self._session = onnxruntime.InferenceSession(f"{local_dir}/onnx/model.onnx")  # type: ignore[attr-defined]

    def _pre_processing(self, input_text: str) -> StandardPreprocessOutput:
        """Tokenize the full input untruncated and split it into overlapping chunks.

        Args:
            input_text: The text to classify.

        Returns:
            GuardrailPreprocessOutput wrapping ``{"chunks": [...]}``, one
            ``{"input_ids", "attention_mask"}`` numpy array pair per chunk, each
            with [CLS]/[SEP] added back and a batch dimension of 1.

        """
        content_ids: list[int] = self._tokenizer(input_text, add_special_tokens=False, truncation=False)["input_ids"]
        cls_id = self._tokenizer.cls_token_id
        sep_id = self._tokenizer.sep_token_id

        chunks: list[AnyDict] = []
        for token_chunk in _chunk_token_ids(content_ids):
            ids = [cls_id, *token_chunk, sep_id]
            chunks.append(
                {
                    "input_ids": np.array([ids], dtype=np.int64),
                    "attention_mask": np.array([[1] * len(ids)], dtype=np.int64),
                }
            )
        return GuardrailPreprocessOutput(data={"chunks": chunks})

    def _inference(self, model_inputs: StandardPreprocessOutput) -> StandardInferenceOutput:
        """Run each chunk through the fused ONNX graph and softmax to P(suspicious).

        Bypasses ``provider.infer()`` entirely: the fused encoder+head graph is run
        directly through ``onnxruntime.InferenceSession.run()`` per chunk.
        """
        required_names = {inp.name for inp in self._session.get_inputs()}
        output_names = [output.name for output in self._session.get_outputs()]
        logits_idx = output_names.index("logits") if "logits" in output_names else 0

        chunk_scores: list[float] = []
        for chunk in model_inputs.data["chunks"]:
            onnx_inputs: AnyDict = {
                "input_ids": chunk["input_ids"],
                "attention_mask": chunk["attention_mask"],
            }
            if "token_type_ids" in required_names:
                onnx_inputs["token_type_ids"] = np.zeros_like(chunk["input_ids"])
            outputs = self._session.run(None, onnx_inputs)
            logits = outputs[logits_idx][0]
            exp_logits = np.exp(logits - np.max(logits))
            probabilities = exp_logits / exp_logits.sum()
            chunk_scores.append(float(probabilities[1]))
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
