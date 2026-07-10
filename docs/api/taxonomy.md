# Taxonomy

The vocabulary behind guardrail metadata (see the [AnyGuardrail reference](any_guardrail.md) for the `list_guardrails` / `group_by` query API and [Guardrails](guardrails/index.md) for the catalog grouped by primary category).

A machine-readable export of every guardrail's metadata is published at <https://raw.githubusercontent.com/mozilla-ai/any-guardrail/main/schemas/guardrail_metadata.json>.

## GuardrailCategory

What a guardrail is designed to detect (a guardrail may span several).

| Value | Meaning |
|-------|---------|
| `prompt_injection` | Prompt injection, jailbreak, and instruction-override attempts. |
| `content_safety` | Harmful content: violence, sexual, self-harm, dangerous, or criminal material. |
| `toxicity` | Hate, harassment, and profanity. |
| `pii` | Personal / sensitive-data detection. |
| `hallucination` | Groundedness / RAG-faithfulness of a response against provided context. |
| `off_topic` | Topical relevance / answer relevance. |
| `bias` | Social bias / fairness. |
| `tool_use` | Function-calling / agent-action validity. |
| `general_judge` | Open-ended rubric / quality scoring against bring-your-own criteria. |

## GuardrailStage

Where in a request/response flow a guardrail runs.

A guardrail that screens both the prompt and the response has ``stages ==
{INPUT, OUTPUT}`` (there is no separate ``EITHER`` value). ``RAG_CONTEXT``
marks guardrails that additionally consume retrieved documents/context.

| Value | Meaning |
|-------|---------|
| `input` | Screens the user prompt (pre-call). |
| `output` | Screens the model response (post-call). |
| `rag_context` | Consumes retrieved documents/context (e.g. groundedness checks). |

## OutputShape

The decision form a guardrail produces (aligns with the populated ``GuardrailOutput`` fields).

| Value | Meaning |
|-------|---------|
| `binary` | A single flagged / not-flagged verdict. |
| `multi_label` | Independent per-category scores/verdicts. |
| `categorical` | A taxonomy verdict (e.g. Llama Guard S-codes). |
| `score` | A scalar risk score. |
| `rubric` | A judge score against a rubric (e.g. 1-5 / 1-10). |
| `span` | Character-offset spans (e.g. hallucination or PII spans). |

## BackendType

How a guardrail executes.

| Value | Meaning |
|-------|---------|
| `local_encoder` | A local encoder classifier (HuggingFace or encoderfile). |
| `local_decoder` | A local decoder LLM (HuggingFace or llamafile). |
| `hosted_api` | A hosted service requiring a key/endpoint. |
| `library_wrapped` | A third-party Python library invoked directly (its own optional extra). |
