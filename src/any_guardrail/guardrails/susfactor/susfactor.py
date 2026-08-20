from typing import Any, ClassVar

from any_guardrail.base import GuardrailName, StandardGuardrail
from any_guardrail.guardrails.utils import default
from any_guardrail.providers.base import StandardProvider
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

SUSFACTOR_ONNX_MODEL_ID = "0dinai/susfactor-e5-large-onnx"
SUSFACTOR_API_MODEL_ID = "susfactor-api"

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
    companion ``onnx/model.onnx_data`` external-data file) in the model repository. By default
    Susfactor bypasses the provider layer and runs the fused graph directly through
    ``onnxruntime.InferenceSession`` — no ``torch``/``transformers`` model classes are involved at
    inference time, only a tokenizer.

    The input text is tokenized untruncated. Sequences that exceed ``MAX_CONTENT_TOKENS`` (510,
    leaving room for [CLS]/[SEP] in the 512-token encoder limit) are split into overlapping chunks
    (chunk size 510, stride 460, 50-token overlap) so no content is silently truncated. Each chunk is
    scored independently; a suspicious verdict on *any* chunk flags the whole input.

    **Hosted backend.** The model repository is gated on HuggingFace, so 0DIN also serves the same
    classifier over REST. Pass ``provider=ZeroDinProvider()`` (from
    ``any_guardrail.providers.zero_din``, credentials via ``api_key=`` or the ``ODIN_API_KEY``
    environment variable) to run against 0DIN Defense instead — same class, same ``threshold``, same
    ``GuardrailOutput``, and no ``onnx`` extra required. Two differences are worth knowing: the
    hosted API chunks server-side with unpublished size/stride parameters and returns only the
    maximum chunk score, so the two backends can disagree on very long inputs; and its own
    ``is_suspicious`` flag is hardcoded at 0.5 and is ignored here — ``threshold`` still decides
    ``valid``, with the untouched response JSON available in ``raw``.

    Any provider supplied here must return ``{"chunk_scores": list[float]}`` from ``infer()``.

    Verdict mapping onto ``GuardrailOutput``:

    - ``score`` (canonical risk: higher = riskier) is the maximum per-chunk ``P(suspicious)`` across
      all chunks.
    - ``valid`` is ``True`` when every chunk's suspicious probability stays below ``threshold``
      (default 0.5).
    - ``categories`` carries a single ``suspicious`` entry with that max score and the overall
      ``triggered`` flag.
    - A backend that returns no scores at all fails closed: ``valid=False`` with
      ``extra={"parse_failure": True}``.

    Expected input: a single ``input_text`` string. There is no prompt+response or chat-message
    mode. List input is accepted via the inherited ``validate()``, which scores each item in turn —
    on the hosted backend that is one HTTP request per item, since the API has no batch endpoint.

    See https://0din.ai/docs/defense/introduction for SusFactor's model card, threat-feed
    training methodology, and benchmark results, and https://0din.ai/docs/defense/api for the
    hosted API.

    The model repository (``0dinai/susfactor-e5-large-onnx``) is gated on HuggingFace; loading it
    requires an authenticated Hub token available via the standard
    ``transformers``/``huggingface_hub`` resolution (environment variable or cached login).

    Args:
        model_id: Optional model ID; must be one of ``SUPPORTED_MODELS``. Defaults to
            ``0dinai/susfactor-e5-large-onnx`` locally, or to whichever entry a supplied
            ``provider`` declares (``susfactor-api`` for the hosted backend).
        threshold: Per-chunk suspicious-probability cutoff at or above which a chunk (and therefore
            the whole input) is flagged. Defaults to 0.5.
        provider: Optional execution provider. ``None`` runs the local ONNX graph; pass
            ``ZeroDinProvider()`` to run against 0DIN's hosted API instead.
        session: Optional pre-built ``onnxruntime.InferenceSession``. Useful for testing: when both
            ``session`` and ``tokenizer`` are supplied, no download or model loading happens.
            Ignored when ``provider`` is set.
        tokenizer: Optional pre-built tokenizer. Useful for testing; see ``session``.

    """

    SUPPORTED_MODELS: ClassVar = [SUSFACTOR_ONNX_MODEL_ID, SUSFACTOR_API_MODEL_ID]

    METADATA: ClassVar[GuardrailMetadata] = GUARDRAIL_METADATA[GuardrailName.SUSFACTOR]

    def __init__(
        self,
        model_id: str | None = None,
        threshold: float = SUSFACTOR_DEFAULT_THRESHOLD,
        provider: StandardProvider | None = None,
        session: Any = None,
        tokenizer: Any = None,
    ) -> None:
        """Initialize the Susfactor guardrail.

        Args:
            model_id: Optional model ID. Must be one of ``SUPPORTED_MODELS``; defaults to
                ``0dinai/susfactor-e5-large-onnx`` locally, or to the ``provider``'s own
                ``default_model_id`` when one is supplied.
            threshold: Per-chunk suspicious-probability cutoff at or above which a chunk (and
                therefore the whole input) is flagged unsafe. Defaults to 0.5.
            provider: Optional execution provider whose ``infer()`` returns
                ``{"chunk_scores": [...]}``. ``None`` runs the local ONNX graph; pass
                ``ZeroDinProvider()`` for 0DIN's hosted API.
            session: Optional pre-built ``onnxruntime.InferenceSession``. If supplied together with
                ``tokenizer``, it is used directly and no download/model loading happens. Ignored
                when ``provider`` is set.
            tokenizer: Optional pre-built tokenizer. If supplied together with ``session``, it is
                used directly and no download/model loading happens.

        Raises:
            ValueError: If ``model_id`` is not in ``SUPPORTED_MODELS``, or if it names the hosted
                service without a matching ``provider``.

        """
        self.threshold = threshold
        self.provider = provider

        if model_id is None:
            # Providers declare which SUPPORTED_MODELS entry they serve, so usage.model_id
            # reports the backend that actually ran rather than the local model's repo id.
            declared = getattr(provider, "default_model_id", None)
            if isinstance(declared, str):
                model_id = declared

        if provider is not None:
            # Resolve before the onnx imports below: an API-only install has no onnxruntime.
            self.model_id = default(model_id, self.SUPPORTED_MODELS)
            provider.load_model(self.model_id)
            return

        if model_id == SUSFACTOR_API_MODEL_ID:
            msg = (
                f"model_id={SUSFACTOR_API_MODEL_ID!r} names 0DIN's hosted service, which needs "
                "provider=ZeroDinProvider(...). Omit model_id to run the local ONNX model."
            )
            raise ValueError(msg)

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

        Delegates to ``provider.pre_process()`` when a provider is configured — 0DIN's
        hosted API tokenizes and chunks server-side, so there is nothing to do locally.

        Args:
            input_text: The text to classify.

        Returns:
            GuardrailPreprocessOutput wrapping ``{"chunks": [...]}``, one
            ``{"input_ids", "attention_mask"}`` numpy array pair per chunk, each
            with [CLS]/[SEP] added back and a batch dimension of 1. When a provider
            is configured, whatever shape that provider's ``pre_process`` returns.

        """
        if self.provider is not None:
            return self.provider.pre_process(input_text)

        # Lazy-import numpy so an API-only install (no `onnx` extra) can still import
        # and run this guardrail through a provider.
        import numpy as np

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

        With no provider this bypasses the provider layer entirely: the fused
        encoder+head graph is run directly through ``onnxruntime.InferenceSession.run()``
        per chunk. With a provider, the call is delegated; either way the result is
        ``{"chunk_scores": [...]}``.
        """
        if self.provider is not None:
            return self.provider.infer(model_inputs)

        import numpy as np

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
        """Reduce per-chunk suspicion probabilities to a single verdict.

        Backend-agnostic: it only needs ``chunk_scores``. ``max(S) >= threshold`` and
        ``any(s >= threshold for s in S)`` are the same predicate, so a backend that
        chunks server-side and returns only the maximum (0DIN's hosted API) produces
        an identical verdict to the local per-chunk list.
        """
        raw = model_outputs.data.get("raw")
        if "chunk_scores" not in model_outputs.data:
            msg = (
                "Susfactor needs a provider whose infer() returns {'chunk_scores': [float, ...]}; "
                f"{type(self.provider).__name__} returned keys {sorted(model_outputs.data)}. "
                "Use ZeroDinProvider for 0DIN's hosted API, or omit provider= to run locally."
            )
            raise RuntimeError(msg)

        chunk_scores: list[float] = model_outputs.data["chunk_scores"]
        if not chunk_scores:
            # No scores at all means no evidence of safety: fail closed rather than
            # invent a verdict (and never call max() on an empty sequence).
            return GuardrailOutput(
                valid=False,
                categories=[CategoryResult(name="suspicious", triggered=True)],
                extra={"parse_failure": True},
                raw=raw,
                usage=GuardrailUsage(model_id=self.model_id),
            )

        score = max(chunk_scores)
        is_suspicious = any(chunk_score >= self.threshold for chunk_score in chunk_scores)
        return GuardrailOutput(
            valid=not is_suspicious,
            score=score,
            categories=[CategoryResult(name="suspicious", triggered=is_suspicious, score=score)],
            raw=raw,
            usage=GuardrailUsage(model_id=self.model_id),
        )
