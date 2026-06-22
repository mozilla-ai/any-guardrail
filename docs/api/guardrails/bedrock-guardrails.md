# BedrockGuardrails

Guardrail wrapping the AWS Bedrock ``ApplyGuardrail`` API.

Provides a uniform interface to AWS Bedrock Guardrails — a unified, FM-agnostic
policy platform covering content moderation, prompt-injection / denied-topic
filtering, PII detection and anonymization, word filters, contextual grounding
(hallucination) checks, and Automated Reasoning policies in a single
configuration. The underlying call is the ``ApplyGuardrail`` action on the
``bedrock-runtime`` boto3 client, which is independent of any foundation model
and can be applied to inputs or outputs.

The most distinctive capability surfaced here is Bedrock's **Automated
Reasoning checks**: an SMT-solver-driven hallucination verification pipeline
that mathematically verifies model outputs against a formal policy. AWS
reports up to 99% verification accuracy for content covered by the policy
[AWS news, 2024-2025]. Policies are authored from natural-language source
documents via auto-extracted schemas. The underlying formal-methods stack
descends from the AWS Automated Reasoning Group's research lineage of 100+
peer-reviewed papers (CAV, POPL, and adjacent venues).

Research and documentation references:
    - AWS blog (GA announcement): Automated Reasoning checks deliver up to
      99% verification accuracy via SMT solvers —
      https://aws.amazon.com/blogs/aws/minimize-ai-hallucinations-and-deliver-up-to-99-verification-accuracy-with-automated-reasoning-checks-now-available/
    - AWS docs: Bedrock Guardrails Automated Reasoning concepts (policy
      authoring, schema extraction, SMT-based verification) —
      https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-automated-reasoning-checks.html
    - AWS docs: Use ApplyGuardrail independently of any FM —
      https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-use-independent-api.html
    - AWS Automated Reasoning Group research lineage (100+ peer-reviewed
      CAV/POPL papers backing the formal-methods techniques).

Authentication uses the standard AWS SigV4 credential chain (environment
variables, IAM role, AWS config file); credentials can also be passed
explicitly or via a custom ``boto3.Session``. The Guardrail itself must be
created out-of-band in the AWS console; the constructor takes its
``guardrail_identifier`` and ``guardrail_version``.

Billing is pay-per-text-unit on the ApplyGuardrail call; Automated Reasoning
checks are a premium tier.

## Supported Models

- `bedrock-guardrails`

## Constructor

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `guardrail_identifier` | `str` | Yes | — |
| `guardrail_version` | `str` | No | `"DRAFT"` |
| `source` | `str` | No | `"INPUT"` |
| `region_name` | `str | None` | No | `None` |
| `aws_access_key_id` | `str | None` | No | `None` |
| `aws_secret_access_key` | `str | None` | No | `None` |
| `boto3_session` | `Any` | No | `None` |

Initialize the AWS Bedrock Guardrails client.

## validate

Default validation pipeline: preprocess -> inference -> postprocess.

**Parameters**

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `input_text` | `str | list[str]` | Yes | — |

**Returns:** `GuardrailOutput | list[GuardrailOutput]`
