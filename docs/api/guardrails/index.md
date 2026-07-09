# Guardrails

Available guardrails, grouped by primary category. Select a guardrail to view its API details.

Query this catalog programmatically with `AnyGuardrail.list_guardrails(...)` and `AnyGuardrail.group_by(...)` — see the [AnyGuardrail reference](../any_guardrail.md).

## Prompt Injection

| Guardrail | Description |
|-----------|-------------|
| [Azure Prompt Shields](azure-prompt-shields.md) | Azure AI Prompt Shields — hosted detector for direct (user prompt) and indirect (document-borne) prompt-injection and jailbreak attacks (Microsoft). |
| [Deepset](deepset.md) | Deepset — binary prompt-injection classifier built on DeBERTa-v3-base (deepset). |
| [Jasper](jasper.md) | Jasper — binary prompt-injection classifiers built on DeBERTa-v3-base and gELECTRA-base (JasperLS). |
| [Lakera Guard](lakera-guard.md) | Lakera Guard — hosted API for prompt-injection, jailbreak, content-moderation, and PII detection (Lakera). |
| [Pangolin Guard](pangolin.md) | Pangolin Guard — binary prompt-injection classifier built on ModernBERT (dcarpintero). |
| [PIGuard](injec-guard.md) | PIGuard — binary prompt-injection classifier built on DeBERTa-v3 and trained to mitigate over-defense (successor to InjecGuard). |
| [Prompt Guard 2](prompt-guard.md) | Prompt Guard 2 — encoder classifier for prompt-injection and jailbreak detection (Meta). |
| [ProtectAI](protectai.md) | ProtectAI — binary prompt-injection classifiers built on DeBERTa-v3 and DistilRoBERTa (ProtectAI). |
| [Sentinel](sentinel.md) | Sentinel — binary prompt-injection classifier built on DeBERTa (Qualifire). |

## Content Safety

| Guardrail | Description |
|-----------|-------------|
| [Alinia](alinia.md) | Alinia — hosted content-moderation and safety-detection API with configurable detection policies (Alinia AI). |
| [Azure Content Safety](azure-content-safety.md) | Azure AI Content Safety — hosted moderation of text and images across hate, sexual, self-harm, and violence categories with 0-7 severity scores (Microsoft). |
| [Bedrock Guardrails](bedrock-guardrails.md) | AWS Bedrock Guardrails — hosted, configurable moderation via the ApplyGuardrail API covering content filters, denied topics, PII, word filters, and contextual grounding (Amazon). |
| [Bielik Guard](bielik-guard.md) | Bielik Guard — Polish multi-label safety classifier (SpeakLeash / Bielik.AI). |
| [DuoGuard](duo-guard.md) | DuoGuard — multilingual multi-label safety classifier scoring text across 12 harm categories including jailbreak prompts. |
| [GLiGuard](gli-guard.md) | GLiGuard — schema-driven safety, toxicity, jailbreak, and refusal detector built on GLiNER2 (Fastino). |
| [gpt-oss-safeguard](gpt-oss-safeguard.md) | gpt-oss-safeguard — policy-grounded reasoning safety classifier that judges text against a bring-your-own written policy (OpenAI). |
| [Granite Guardian](granite-guardian.md) | Granite Guardian — hybrid-thinking safety and judge model covering harm, RAG groundedness, and function-calling risks via bring-your-own-criteria (IBM). |
| [HarmAug-Guard](harm-guard.md) | HarmAug-Guard — binary safety and jailbreak classifier built on DeBERTa-v3-large, scoring a prompt or prompt-response pair. |
| [Kanana Safeguard](kanana-safeguard.md) | Kanana Safeguard — Korean safety decoder models covering harmful content, legal risk, and prompt attacks (Kakao). |
| [Llama Guard](llama-guard.md) | Llama Guard — decoder-LLM safety classifier judging prompts and responses against the 14-category MLCommons hazard taxonomy (Meta). |
| [Nemotron Content Safety](nemotron-content-safety.md) | Nemotron Content Safety — 4B reasoning safety classifier covering a 22-category content-safety taxonomy (NVIDIA). |
| [OpenAI Moderation](openai-moderation.md) | OpenAI Moderation — hosted moderation API flagging content across 13 harm categories with calibrated scores (OpenAI). |
| [PolyGuard](poly-guard.md) | PolyGuard — multilingual safety-moderation judge reporting request harm, response harm, and refusal across 17 languages. |
| [Qwen3Guard](qwen3-guard.md) | Qwen3Guard-Gen — generative safety moderation with three-level severity across 119 languages (Qwen). |
| [Qwen3Guard-Stream](qwen3-guard-stream.md) | Qwen3Guard-Stream — token-level streaming safety moderation with span output (Qwen). |
| [ShieldGemma](shield-gemma.md) | ShieldGemma — policy-conditioned safety classifier that judges a prompt against a user-supplied policy via Yes/No token logits (Google). |
| [watsonx Guardian](watsonx-guardian.md) | watsonx Guardian — hosted text-detection moderation API running configurable Granite Guardian detectors (IBM). |
| [WildGuard](wild-guard.md) | WildGuard — one-pass safety-moderation judge reporting prompt harm, response harm, and refusal (Allen Institute for AI). |

## Hallucination

| Guardrail | Description |
|-----------|-------------|
| [LettuceDetect](lettuce-detect.md) | LettuceDetect — token/span-level RAG hallucination detector built on ModernBERT (KRLabs). |
| [Patronus](patronus.md) | Patronus — hosted evaluation API running configurable evaluators for hallucination, toxicity, PII, prompt injection, and custom judging (Patronus AI). |

## Off-Topic

| Guardrail | Description |
|-----------|-------------|
| [Off-Topic](off-topic.md) | Off-Topic — cross-encoder relevance detector that flags whether an input strays from a comparison text (GovTech Singapore). |

## General Judge

| Guardrail | Description |
|-----------|-------------|
| [AnyLlm](any-llm.md) | AnyLlm — policy-based LLM judge that grades text against a natural-language policy using any LLM provider supported by any-llm. |
| [CompassJudger](compass-judger.md) | CompassJudger — generalist LLM judge that scores a response against user-defined criteria and rubric on a 1-10 scale (OpenCompass). |
| [DynaGuard](dyna-guard.md) | DynaGuard — dynamic guardian model evaluating conversation compliance with user-defined policies. |
| [Flow Judge](flowjudge.md) | Flow Judge — local LLM judge scoring text against user-defined criteria, metrics, and rubrics via the flow-judge library (Flow AI). |
| [GLIDER](glider.md) | GLIDER — prompt-based LLM judge that grades text against user-supplied pass criteria and rubric, returning reasoning and highlighted phrases (Patronus AI). |
| [Prometheus](prometheus.md) | Prometheus — open rubric-based LLM judge grading a response on a user-defined 1-5 rubric (KAIST / prometheus-eval). |
| [Selene 1 Mini](selene.md) | Selene 1 Mini — general-purpose LLM judge grading a response against a user-defined 1-5 rubric (Atla). |
