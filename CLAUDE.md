# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`any-guardrail` is a unified Python library (Python 3.11+) providing a consistent interface for multiple AI safety guardrail backends. It lets developers swap between guardrail implementations — and the runtimes that execute them — without changing application code. Maintained by Mozilla AI under Apache 2.0.

Two orthogonal layers:

- **Guardrails** (`src/any_guardrail/guardrails/<name>/<name>.py`) — domain logic. Each one defines what to check (harm, prompt-injection, off-topic, etc.) and how to interpret model output. Every guardrail exposes a `validate(input_text, ...)` method returning a `GuardrailOutput`.
- **Providers** (`src/any_guardrail/providers/<name>.py`) — execution backends. They load models, tokenize, and run inference. Guardrails accept a `provider=` kwarg, so the same guardrail can run against different backends (e.g. local HuggingFace transformers via `HuggingFaceProvider`, Mozilla single-binary classifiers via `EncoderfileProvider`, or Mozilla single-binary decoder LLMs via `LlamafileProvider`).

## Common Commands

```bash
# Setup
uv venv && source .venv/bin/activate
uv sync --dev --extra all

# Unit tests (fast, ~2s for the whole suite)
pytest -v tests/unit

# Single test file / single test by name
pytest -v tests/unit/test_api.py
pytest -v tests/unit/test_api.py -k "test_name"

# Integration tests (downloads real models; some need API keys
#   like OPENAI_API_KEY, CONTENT_SAFETY_KEY, CONTENT_SAFETY_ENDPOINT, ALINIA_API_KEY)
pytest -v tests/integration -n auto

# Lint, format, type-check (runs ruff, mypy strict, codespell, nbstripout)
pre-commit run --all-files

# Live docs preview (mkdocs)
mkdocs serve

# Generate API reference pages from docstrings (writes into docs/api/ or site/api/)
python scripts/generate_api_docs.py
```

## Architecture

### Core abstractions (`src/any_guardrail/`)

- **`base.py`** — class hierarchy:
  - `Guardrail` (ABC) — base class; every guardrail implements `validate()` returning a `GuardrailOutput[ValidT, ExplanationT, ScoreT]`.
  - `ThreeStageGuardrail` (ABC) — opinionated pipeline subclass: `_pre_processing` → `_inference` → `_post_processing`. The default `validate()` chains these and supports list inputs (batched).
  - `GuardrailName` — string enum of all supported guardrail identifiers (used by the factory).
  - `StandardGuardrail` — type alias for `ThreeStageGuardrail[AnyDict, AnyDict, bool, None, float]`, used by simple binary classifiers.
- **`types.py`** — generic Pydantic models (`GuardrailOutput`, `GuardrailPreprocessOutput`, `GuardrailInferenceOutput`) plus shared aliases (`AnyDict`, `ChatMessages`, `BinaryScoreOutput`, etc.). All wrappers use `arbitrary_types_allowed=True` so numpy arrays / torch tensors pass through.
- **`api.py`** — `AnyGuardrail` factory. `AnyGuardrail.create(GuardrailName.PROTECTAI, ...)` does dynamic module import using the convention: snake_case enum value → `any_guardrail.guardrails.<name>.<name>` module → PascalCase class.

### Providers (`src/any_guardrail/providers/`)

Providers are an execution layer that guardrails compose with, not inherit from.

- **`base.py`** — `Provider` (ABC) with abstract `load_model()`, `pre_process()`, `infer()` methods. `StandardProvider` is the type alias `Provider[AnyDict, AnyDict]`.
- **`huggingface.py`** — `HuggingFaceProvider` is the default for most guardrails. Loads via `transformers.from_pretrained`, tokenizes per-call, runs `model(**inputs)` under `torch.no_grad()`. Surfaces `device`, `torch_dtype`, `cache_dir`, `revision`, `model_kwargs`, `tokenizer_kwargs`, `multi_label` as constructor args.
- **`encoderfile.py`** — `EncoderfileProvider` runs Mozilla AI's [`encoderfile`](https://github.com/mozilla-ai/encoderfile) single-binary format (encoder + classification head packaged as one executable). On `load_model()` it auto-downloads the platform-specific artifact from `mozilla-ai/encoderfile` on HuggingFace, spawns the binary as a subprocess HTTP server, polls `/predict` for readiness, then proxies inference calls over `127.0.0.1`. Drop-in for `HuggingFaceProvider` on supported encoder classifiers — no `torch`/`transformers` install required. Implements the context manager protocol (`with EncoderfileProvider() as provider: ...`) for deterministic subprocess cleanup; `atexit` is the fallback. Curated artifact map lives at `providers/_encoderfile_artifacts.py`. macOS + Linux only.
- **`llamafile.py`** — `LlamafileProvider` runs Mozilla's [`llamafile`](https://github.com/Mozilla-Ocho/llamafile) format (decoder LLM = llama.cpp + GGUF weights packaged via Cosmopolitan Libc as one multi-platform executable). On `load_model()` it auto-downloads the artifact, spawns it as an OpenAI-compatible HTTP server (`--server --jinja --no-webui`), polls `/health` for readiness, then routes `generate_chat()` calls to `POST /v1/chat/completions`. **Chat-only** — `pre_process()` and `infer()` raise `NotImplementedError`; decoder-LLM guardrails consume it via `generate_chat()` instead. Implements the context manager protocol (`with LlamafileProvider() as provider: ...`) for deterministic subprocess cleanup; `atexit` is the fallback. Two platform notes: launching is done via `sh <binary>` because macOS arm64 can't `execve` Cosmopolitan APE polyglots directly (the APE prelude is valid POSIX shell that exec's into the binary); GPU offload is exposed via `n_gpu_layers=`. Curated artifact map at `providers/_llamafile_artifacts.py`; constructor also accepts `repo_id=` + `filename=` overrides.

A guardrail's `__init__` typically builds a default `HuggingFaceProvider` if none is supplied:

```python
class MyGuardrail(StandardGuardrail):
    def __init__(self, model_id=None, provider=None):
        self.model_id = default(model_id, self.SUPPORTED_MODELS)
        self.provider = provider or HuggingFaceProvider()
        self.provider.load_model(self.model_id)
```

For decoder-LLM-backed guardrails (e.g. `GraniteGuardian`, `LlamaGuard`, `ShieldGemma`), the default provider passes the right `model_class`/`tokenizer_class` for the model. If you construct a `HuggingFaceProvider()` yourself and pass it to those guardrails, you must pass the same — otherwise `from_pretrained` will reject the config.

#### Uniform `infer()` shape

Both providers return the same dict shape from `infer()` so guardrails are provider-agnostic at the post-processing stage:

- `logits`: per-input logits (numpy array, shape `(batch, num_classes)`).
- `scores`: softmax or sigmoid (when `multi_label=True`) of logits.
- `predicted_indices`: list of argmax indices, one per row.
- `predicted_labels`: list of labels resolved via `id2label` (HF) or returned natively by the binary (encoderfile).
- `labels`: full ordered label list aligning index-wise with `scores` columns (HF via `id2label`; `None` for encoderfile, whose `/predict` response only carries the predicted label — 2-class guardrails fall back to complement logic).

Exception: causal-LM-backed guardrails using `HuggingFaceProvider` (`ShieldGemma`) produce 3D logits `(batch, seq, vocab)`. In that case `infer()` returns `logits` as a raw torch tensor and sets `scores`/`predicted_indices`/`predicted_labels` to `None`; the guardrail does its own selection (e.g. `logits[0, -1, [vocab["Yes"], vocab["No"]]]`) and softmax. This is checked in unit tests — when adding a new guardrail that runs a causal LM through `provider.infer()`, expect the raw-tensor path.

#### Opt-in `generate_chat()` for chat-style decoder LLMs

`Provider.generate_chat(messages, max_new_tokens, ...)` is the parallel contract for decoder-LLM workflows that don't fit the `pre_process` → `infer` shape (e.g. judge models that emit a yes/no string). The base implementation raises `NotImplementedError` so encoder providers (`EncoderfileProvider`, `AzureContentSafety`) are unaffected — it's opt-in.

Both `HuggingFaceProvider` and `LlamafileProvider` implement it and return the same dict shape:

- `generated_text`: decoded new tokens only (prompt stripped).
- `prompt_token_count` / `completion_token_count`: ints, or `None` when the backend doesn't surface them (llamafile tokenizes server-side).
- `raw`: provider-specific raw output (HF tensor / OpenAI JSON), for callers that need it.

`GraniteGuardian` and `LlamaGuard` consume `generate_chat()` instead of touching `provider.tokenizer.apply_chat_template` + `provider.model.generate` directly, which is what lets them swap between the HF and llamafile backends. They no longer `import torch`. `chat_template_kwargs` (e.g. RAG `documents`, `available_tools`) and `generation_kwargs` (e.g. `pad_token_id` for Llama Guard 3) are pass-throughs.

### Guardrail implementations (`src/any_guardrail/guardrails/`)

Each guardrail lives in its own subdirectory (e.g. `llama_guard/llama_guard.py`). They inherit from `Guardrail`, `ThreeStageGuardrail`, or `StandardGuardrail` depending on shape:

- `Guardrail` directly: API-based or fully custom shape — `AnyLlm` (any-llm SDK), `Alinia` (HTTP API).
- `ThreeStageGuardrail` with custom generics: generative/judge models or non-binary outputs — `GraniteGuardian`, `LlamaGuard`, `Glider`, `Flowjudge`, `AzureContentSafety`, `DuoGuard` (multi-label), `OffTopic`.
- `StandardGuardrail`: simple binary classifiers — `Protectai`, `Deepset`, `Jasper`, `Sentinel`, `Pangolin`, `InjecGuard`, `HarmGuard`.

Shared helpers live in `src/any_guardrail/guardrails/utils.py` (`default()` for model-id validation, `match_injection_label()` for the common "is the predicted label X?" post-processing).

### Adding a new guardrail

1. Create `src/any_guardrail/guardrails/<name>/<name>.py`.
2. Inherit from `Guardrail`, `ThreeStageGuardrail`, or `StandardGuardrail`.
3. Set `SUPPORTED_MODELS: ClassVar = [...]`.
4. Add an entry to `GuardrailName` in `src/any_guardrail/base.py`.
5. In `__init__`, accept `provider: StandardProvider | None = None` and build a default `HuggingFaceProvider` (with the right `model_class`/`tokenizer_class` for the model) if not supplied. Call `self.provider.load_model(self.model_id)`. For decoder-LLM guardrails, do the `from transformers import ...` *inside* the `provider is None` branch (lazy import) so users on a `[llamafile]`-only install can supply their own provider without paying the import cost or hitting `ImportError` at module load; see `granite_guardian.py` / `llama_guard.py` for the pattern.
6. Implement `_pre_processing`, `_inference`, `_post_processing` (or override `validate` directly for non-three-stage shapes).
7. Add an entry to the `GUARDRAILS` list in `scripts/generate_api_docs.py` so the API reference page is generated. Add a link to it from `docs/SUMMARY.md`.
8. Tests: unit tests under `tests/unit/` (mock the provider so weights aren't downloaded); integration tests under `tests/integration/` (real model, `@pytest.mark.skipif(RUNNING_IN_CI, ...)` if the model is large).

The factory auto-discovers via the naming convention: snake_case directory name in `GuardrailName` → matching module → PascalCase class.

## Code quality

- **Ruff** — line length 120, `extend-select = ["ALL"]` with targeted ignores in `pyproject.toml`. Lazy imports are intentional (`PLC0415` is ignored). The pre-commit config pins a specific ruff version (currently 0.15.12); CI runs that pinned version, so always run `pre-commit run --all-files` before pushing.
  - **Watch out:** `pre-commit run` reuses a cached virtualenv per hook revision. When `.pre-commit-config.yaml` bumps the ruff `rev`, your local cache can stay on the old version until you clean it. If CI's linter fails with ruff rules that your local pre-commit didn't flag, run `pre-commit clean && pre-commit run --all-files` to force a fresh install of the pinned version, then re-test. Cached venvs live under `~/.cache/pre-commit/repo*/py_env-*/bin/ruff` — confirm versions with `find ~/.cache/pre-commit -name ruff -type f -exec {} --version \;`.
- **mypy** — strict mode globally (`disallow_untyped_defs`, `disallow_untyped_calls`). Tests are slightly relaxed (`disallow_untyped_decorators=false`) so pytest fixtures don't fight the type checker.
- **codespell, nbstripout, standard hooks** — run via pre-commit.
- **Pytest** — 120s timeout per test (`pyproject.toml`). Integration tests use `pytest-xdist` for parallelism (`-n auto`). Use `RUNNING_IN_CI = os.environ.get("CI") == "true"` to gate model-heavy tests.

## Dependencies

Core deps are minimal: `any-llm-sdk` and `pydantic`. Heavy backends are optional extras:

- `huggingface` — `transformers`, `torch`, `huggingface-hub`, `hf-xet` (used by most guardrails).
- `encoderfile` — `huggingface-hub`, `hf-xet`, `numpy` (no torch/transformers; just enough to download the binary and parse JSON over HTTP).
- `llamafile` — `huggingface-hub`, `hf-xet` (no torch/transformers/numpy; just enough to download the binary and POST chat completions over HTTP).
- `azure-content-safety` — `azure-ai-contentsafety`, `pathvalidate`.
- `flowjudge` — `flow-judge[hf]`.
- `all` — the aggregate extra that pulls them all in.

Pre-commit hooks live in their own `lint` group (in `[dependency-groups]`), pytest tooling in `tests`, docs in `docs`. `uv sync --dev` pulls all three.

## Docs publishing

The docs site is split between mkdocs (local dev) and GitBook (production):

- `docs/cookbook/*.ipynb` — Jupyter notebooks; `nbstripout` enforces clean outputs.
- `docs/api/**.md` — mkdocstrings stubs used by `mkdocs serve` for local preview.
- `scripts/generate_api_docs.py` — generates fully-rendered API reference pages from docstrings; `scripts/convert_to_gitbook.py` uses its output to build the GitBook site (which excludes `docs/api/**` to avoid double rendering).
- `docs/SUMMARY.md` — GitBook navigation. Any new cookbook or API page must be linked from here to appear in production navigation.
