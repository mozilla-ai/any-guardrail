# Guardrails

Available guardrails, grouped by primary category. Select a guardrail to view its API details. See the [Taxonomy reference](../taxonomy.md) for what each category means.

Query this catalog programmatically with `AnyGuardrail.list_guardrails(...)` and `AnyGuardrail.group_by(...)` — see the [AnyGuardrail reference](../any_guardrail.md).

## Prompt Injection

| Guardrail | Description |
|-----------|-------------|
| [Azure Prompt Shields](azure-prompt-shields.md) | Hosted detector for direct (user prompt) and indirect (document-borne) prompt-injection and jailbreak attacks. |
| [Deepset](deepset.md) | Binary prompt-injection classifier. |
| [HarmAug-Guard](harm-guard.md) | Binary safety and jailbreak classifier, scoring a prompt or prompt-response pair. |
| [Jasper](jasper.md) | Binary prompt-injection classifiers. |
| [Lakera Guard](lakera-guard.md) | Hosted API for prompt-injection, jailbreak, content-moderation, and PII detection. |
| [Pangolin Guard](pangolin.md) | Binary prompt-injection classifier. |
| [PIGuard](injec-guard.md) | Binary prompt-injection classifier trained to mitigate over-defense. |
| [Prompt Guard 2](prompt-guard.md) | Encoder classifier for prompt-injection and jailbreak detection. |
| [ProtectAI](protectai.md) | Binary prompt-injection classifiers. |
| [Sentinel](sentinel.md) | Binary prompt-injection classifier. |

## Content Safety

| Guardrail | Description |
|-----------|-------------|
| [Alinia](alinia.md) | Hosted content-moderation and safety-detection API with configurable detection policies. |
| [Azure Content Safety](azure-content-safety.md) | Hosted moderation of text and images across hate, sexual, self-harm, and violence categories with 0-7 severity scores. |
| [Bedrock Guardrails](bedrock-guardrails.md) | Hosted, configurable moderation covering content filters, denied topics, PII, word filters, and contextual grounding. |
| [Bielik Guard](bielik-guard.md) | Polish multi-label safety classifier. |
| [DuoGuard](duo-guard.md) | Multilingual multi-label safety classifier scoring text across 12 harm categories including jailbreak prompts. |
| [GLiGuard](gli-guard.md) | Schema-driven safety, toxicity, jailbreak, and refusal detector. |
| [gpt-oss-safeguard](gpt-oss-safeguard.md) | Policy-grounded reasoning safety classifier that judges text against a bring-your-own written policy. |
| [Kanana Safeguard](kanana-safeguard.md) | Korean safety decoder models covering harmful content, legal risk, and prompt attacks. |
| [Llama Guard](llama-guard.md) | Decoder-LLM safety classifier judging prompts and responses against the 14-category MLCommons hazard taxonomy. |
| [Nemotron Content Safety](nemotron-content-safety.md) | Reasoning safety classifier covering a 22-category content-safety taxonomy. |
| [OpenAI Moderation](openai-moderation.md) | Hosted moderation API flagging content across 13 harm categories with calibrated scores. |
| [PolyGuard](poly-guard.md) | Multilingual safety-moderation judge reporting request harm, response harm, and refusal across 17 languages. |
| [Qwen3Guard](qwen3-guard.md) | Generative safety moderation with three-level severity across 119 languages. |
| [Qwen3Guard-Stream](qwen3-guard-stream.md) | Token-level streaming safety moderation with span output. |
| [ShieldGemma](shield-gemma.md) | Policy-conditioned safety classifier that judges a prompt against a user-supplied policy via Yes/No token logits. |
| [watsonx Guardian](watsonx-guardian.md) | Hosted text-detection moderation API running configurable Granite Guardian detectors. |
| [WildGuard](wild-guard.md) | One-pass safety-moderation judge reporting prompt harm, response harm, and refusal. |

## PII

| Guardrail | Description |
|-----------|-------------|
| [GLiNER2 PII](gli-ner-pii.md) | Span-level PII/NER detector emitting character spans and a redacted copy of the text. |

## Hallucination

| Guardrail | Description |
|-----------|-------------|
| [LettuceDetect](lettuce-detect.md) | Token/span-level RAG hallucination detector. |

## Off-Topic

| Guardrail | Description |
|-----------|-------------|
| [Off-Topic](off-topic.md) | Cross-encoder relevance detector that flags whether an input strays from a comparison text. |

## General Judge

| Guardrail | Description |
|-----------|-------------|
| [AnyLlm](any-llm.md) | Policy-based LLM judge that grades text against a natural-language policy using an LLM provider. |
| [CompassJudger](compass-judger.md) | Generalist LLM judge that scores a response against user-defined criteria and rubric on a 1-10 scale. |
| [DynaGuard](dyna-guard.md) | Dynamic guardian model evaluating conversation compliance with user-defined policies. |
| [Flow Judge](flowjudge.md) | Local LLM judge scoring text against user-defined criteria, metrics, and rubrics. |
| [GLIDER](glider.md) | Prompt-based LLM judge that grades text against user-supplied pass criteria and rubric, returning reasoning and highlighted phrases. |
| [Granite Guardian](granite-guardian.md) | Hybrid-thinking safety and judge model covering harm, RAG groundedness, and function-calling risks via bring-your-own-criteria. |
| [Patronus](patronus.md) | Hosted evaluation API running configurable evaluators for hallucination, toxicity, PII, prompt injection, and custom judging. |
| [Prometheus](prometheus.md) | Open rubric-based LLM judge grading a response on a user-defined 1-5 rubric. |
| [Selene 1 Mini](selene.md) | General-purpose LLM judge grading a response against a user-defined 1-5 rubric. |
