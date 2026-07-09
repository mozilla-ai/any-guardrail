from typing import Any, ClassVar

from any_guardrail.base import GuardrailOutput, ThreeStageGuardrail
from any_guardrail.types import AnyDict, GuardrailInferenceOutput, GuardrailPreprocessOutput

_VALID_SOURCES = ("INPUT", "OUTPUT")


class BedrockGuardrails(ThreeStageGuardrail[AnyDict, AnyDict]):
    """AWS Bedrock Guardrails — hosted, configurable moderation via the ApplyGuardrail API covering content filters, denied topics, PII, word filters, and contextual grounding (Amazon).

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

    """

    SUPPORTED_MODELS: ClassVar = ["bedrock-guardrails"]

    def __init__(
        self,
        guardrail_identifier: str,
        guardrail_version: str = "DRAFT",
        source: str = "INPUT",
        region_name: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        boto3_session: Any = None,
    ) -> None:
        """Initialize the AWS Bedrock Guardrails client.

        Args:
            guardrail_identifier: The AWS guardrail identifier (ID or ARN) of an
                existing Bedrock Guardrail. Create one in the Bedrock console
                before instantiating this class.
            guardrail_version: The guardrail version to apply. ``"DRAFT"`` refers
                to the working draft; otherwise pass a published numeric version
                string (e.g. ``"1"``).
            source: Either ``"INPUT"`` (apply policy to a user prompt) or
                ``"OUTPUT"`` (apply policy to a model response). Validated on
                construction.
            region_name: Optional AWS region (e.g. ``"us-east-1"``). Falls back
                to the boto3 default chain when ``None``.
            aws_access_key_id: Optional explicit access key. Prefer environment
                variables or an IAM role in production.
            aws_secret_access_key: Optional explicit secret key. Prefer
                environment variables or an IAM role in production.
            boto3_session: Optional pre-configured ``boto3.Session`` instance.
                When supplied, takes precedence over the explicit credentials
                and region_name kwargs.

        Raises:
            ValueError: If ``source`` is not ``"INPUT"`` or ``"OUTPUT"``.
            ImportError: If the ``boto3`` package is not installed.

        """
        if source not in _VALID_SOURCES:
            msg = f"source must be one of {_VALID_SOURCES}, got {source!r}"
            raise ValueError(msg)

        try:
            import boto3
        except ImportError as e:
            msg = (
                "boto3 package is not installed. "
                "Please install it with `pip install 'any-guardrail[bedrock]'` to use BedrockGuardrails."
            )
            raise ImportError(msg) from e

        self.guardrail_identifier = guardrail_identifier
        self.guardrail_version = guardrail_version
        self.source = source

        self.client: Any
        if boto3_session is not None:
            self.client = boto3_session.client("bedrock-runtime")
        else:
            self.client = boto3.client(  # type: ignore[no-untyped-call]
                "bedrock-runtime",
                region_name=region_name,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
            )

    def _pre_processing(self, text: str) -> GuardrailPreprocessOutput[AnyDict]:
        """Wrap ``text`` in the ApplyGuardrail ``content`` payload shape.

        Args:
            text: The text to evaluate against the guardrail policy.

        Returns:
            A preprocessing output whose ``data`` dict contains the
            ``content`` array consumed by ``client.apply_guardrail``.

        """
        return GuardrailPreprocessOutput(data={"content": [{"text": {"text": text}}]})

    def _inference(self, model_inputs: GuardrailPreprocessOutput[AnyDict]) -> GuardrailInferenceOutput[AnyDict]:
        """Call the Bedrock ``ApplyGuardrail`` API and return the raw response.

        Args:
            model_inputs: Preprocessing output with the ``content`` payload.

        Returns:
            An inference output wrapping the full boto3 response dict (includes
            ``action``, ``outputs``, ``assessments``, and usage metadata).

        """
        response = self.client.apply_guardrail(
            guardrailIdentifier=self.guardrail_identifier,
            guardrailVersion=self.guardrail_version,
            source=self.source,
            content=model_inputs.data["content"],
        )
        return GuardrailInferenceOutput(data=response)

    def _post_processing(self, model_outputs: GuardrailInferenceOutput[AnyDict]) -> GuardrailOutput:
        """Translate the ApplyGuardrail response into a ``GuardrailOutput``.

        Args:
            model_outputs: The wrapped boto3 response dict.

        Returns:
            ``valid=True`` when ``action == "NONE"``; otherwise ``valid=False``.
            ``action`` carries Bedrock's native verdict (e.g. ``"NONE"`` or
            ``"GUARDRAIL_INTERVENED"``). ``extra`` contains the full
            ``assessments`` list (per-policy breakdown: topic, content, word,
            sensitive-information, contextual grounding, and Automated
            Reasoning) and the modified ``outputs``; the untouched boto3
            response remains reachable via ``raw``. ``score`` is ``1.0`` when
            the guardrail intervened and ``0.0`` when no intervention occurred —
            a simple severity proxy for the binary action.

        """
        response = model_outputs.data
        action = response.get("action", "NONE")
        valid = action == "NONE"
        score = 0.0 if valid else 1.0
        return GuardrailOutput(
            valid=valid,
            action=action,
            score=score,
            extra={
                "assessments": response.get("assessments", []),
                "outputs": response.get("outputs", []),
            },
            raw=response,
        )
