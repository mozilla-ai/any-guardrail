# BedrockGuardrails

Hosted, configurable moderation covering content filters, denied topics, PII, word filters, and contextual grounding.

Provides a uniform interface to AWS Bedrock Guardrails — a unified, FM-agnostic
policy platform covering content moderation, prompt-injection / denied-topic
filtering, PII detection and anonymization, word filters, contextual grounding
(hallucination) checks, and Automated Reasoning policies in a single
configuration. The underlying call is the ``ApplyGuardrail`` action on the
``bedrock-runtime`` boto3 client, which is independent of any foundation model
and can be applied to inputs or outputs.

The policy itself lives in AWS: create a Guardrail in the Bedrock console (its
content filters, denied topics, PII entities, word lists, and grounding
thresholds are all configured there), then hand its ``guardrail_identifier``
and ``guardrail_version`` to this class. The ``source`` set on the constructor
decides whether the screened text is treated as a user ``"INPUT"`` prompt or a
model ``"OUTPUT"`` response; there is no separate prompt-vs-response argument on
the call itself.

The most distinctive capability surfaced here is Bedrock's **Automated
Reasoning checks**: an SMT-solver-driven hallucination verification pipeline
that mathematically verifies model outputs against a formal policy. AWS
reports up to 99% verification accuracy for content covered by the policy
[AWS news, 2024-2025]. Policies are authored from natural-language source
documents via auto-extracted schemas. The underlying formal-methods stack
descends from the AWS Automated Reasoning Group's research lineage of 100+
peer-reviewed papers (CAV, POPL, and adjacent venues).

Expected input: a single string (or a list of strings, screened one at a time
via the batched ``ThreeStageGuardrail.validate``).

``GuardrailOutput`` mapping:
    - ``valid`` is ``True`` when Bedrock's ``action`` is ``"NONE"`` (no policy
      intervened); ``False`` otherwise (e.g. ``"GUARDRAIL_INTERVENED"``).
    - ``action`` carries Bedrock's native verdict string.
    - ``score`` is a binary severity proxy for that action: ``1.0`` when the
      guardrail intervened, ``0.0`` when it did not (higher = riskier). Bedrock
      returns a categorical action rather than a continuous risk probability.
    - ``categories`` and ``spans`` are not populated. The full per-policy
      breakdown (topic, content, word, sensitive-information, contextual
      grounding, and Automated Reasoning assessments) is preserved in
      ``extra["assessments"]``, the modified / anonymized text in
      ``extra["outputs"]``, and the untouched boto3 response in ``raw``.

Authentication uses the standard AWS SigV4 credential chain (environment
variables, IAM role, AWS config file); credentials can also be passed
explicitly or via a custom ``boto3.Session``.

Billing is pay-per-text-unit on the ApplyGuardrail call; Automated Reasoning
checks are a premium tier.

For more information, see:

- [Use ApplyGuardrail independently of any FM](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-use-independent-api.html)
- [Automated Reasoning checks (policy authoring, schema extraction, SMT-based verification)](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-automated-reasoning-checks.html)
- [AWS blog: minimize AI hallucinations with Automated Reasoning checks (up to 99% verification accuracy)](https://aws.amazon.com/blogs/aws/minimize-ai-hallucinations-and-deliver-up-to-99-verification-accuracy-with-automated-reasoning-checks-now-available/)

## Supported Models

- `bedrock-guardrails`

## Constructor

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `guardrail_identifier` | `str` | Yes | — | The AWS guardrail identifier (ID or ARN) of an existing Bedrock Guardrail. Create one in the Bedrock console before instantiating this class. |
| `guardrail_version` | `str` | No | `"DRAFT"` | The guardrail version to apply. ``"DRAFT"`` refers to the working draft; otherwise pass a published numeric version string (e.g. ``"1"``). |
| `source` | `str` | No | `"INPUT"` | Either ``"INPUT"`` (apply policy to a user prompt) or ``"OUTPUT"`` (apply policy to a model response). Validated on construction. |
| `region_name` | `str | None` | No | `None` | Optional AWS region (e.g. ``"us-east-1"``). Falls back to the boto3 default chain when ``None``. |
| `aws_access_key_id` | `str | None` | No | `None` | Optional explicit access key. Prefer environment variables or an IAM role in production. |
| `aws_secret_access_key` | `str | None` | No | `None` | Optional explicit secret key. Prefer environment variables or an IAM role in production. |
| `boto3_session` | `Any` | No | `None` | Optional pre-configured ``boto3.Session`` instance. When supplied, takes precedence over the explicit credentials and region_name kwargs. |

Initialize the AWS Bedrock Guardrails client.

## validate

Default validation pipeline: preprocess -> inference -> postprocess.

**Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `input_text` | `str | list[str]` | Yes | — | The text to validate. If a list is supplied, each item is validated and a list of GuardrailOutputs is returned in the same order. Subclasses can override ``_validate_batch`` to enable true batched inference; the default iterates over inputs. |

**Returns:** `GuardrailOutput | list[GuardrailOutput]`
