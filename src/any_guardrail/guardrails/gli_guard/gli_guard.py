import time
from typing import Any, ClassVar

try:
    from gliner2 import GLiNER2

    MISSING_PACKAGES_ERROR = None
except ImportError as e:
    MISSING_PACKAGES_ERROR = e

from any_guardrail.base import Guardrail
from any_guardrail.guardrails.utils import default
from any_guardrail.types import CategoryResult, GuardrailOutput, GuardrailUsage

# GLiGuard schema (task name -> labels), from the model card examples.
SAFETY_LABELS = ["safe", "unsafe"]
TOXICITY_LABELS = [
    "violence_and_weapons",
    "non_violent_crime",
    "sexual_content",
    "hate_and_discrimination",
    "self_harm_and_suicide",
    "pii_exposure",
    "misinformation",
    "copyright_violation",
    "child_safety",
    "political_manipulation",
    "unethical_conduct",
    "regulated_advice",
    "privacy_violation",
    "other",
    "benign",
]
JAILBREAK_LABELS = [
    "prompt_injection",
    "jailbreak_attempt",
    "policy_evasion",
    "instruction_override",
    "system_prompt_exfiltration",
    "data_exfiltration",
    "roleplay_bypass",
    "hypothetical_bypass",
    "obfuscated_attack",
    "multi_step_attack",
    "social_engineering",
    "benign",
]
REFUSAL_LABELS = ["refusal", "compliance"]

GLIGUARD_SCHEMA: dict[str, Any] = {
    "prompt_safety": SAFETY_LABELS,
    "prompt_toxicity": {"labels": TOXICITY_LABELS, "multi_label": True, "cls_threshold": 0.4},
    "jailbreak_detection": {"labels": JAILBREAK_LABELS, "multi_label": True},
    "response_refusal": REFUSAL_LABELS,
}


class GliGuard(Guardrail):
    """GLiGuard (Fastino) — schema-driven safety / toxicity / jailbreak / refusal detector.

    Wraps the ``gliner2`` library to run a 300M encoder that classifies a text across
    four tasks: prompt safety (safe/unsafe), prompt toxicity (15 categories),
    jailbreak detection (12 attack types), and response refusal. ``valid`` is ``True``
    when prompt safety is not ``unsafe`` and no jailbreak category fires. Triggered
    toxicity and jailbreak labels are surfaced in ``categories``.

    For more information, see the
    [gliguard-LLMGuardrails-300M model card](https://huggingface.co/fastino/gliguard-LLMGuardrails-300M).

    Args:
        model_id: Optional HuggingFace model ID. Defaults to ``fastino/gliguard-LLMGuardrails-300M``.
        threshold: Classification threshold passed to ``classify_text``. Defaults to 0.5.

    Raises:
        ImportError: When the ``gliner`` extra is not installed.
    """

    SUPPORTED_MODELS: ClassVar = ["fastino/gliguard-LLMGuardrails-300M"]

    def __init__(self, model_id: str | None = None, threshold: float = 0.5) -> None:
        """Initialize the GLiGuard guardrail."""
        if MISSING_PACKAGES_ERROR is not None:
            msg = "Missing packages for GLiGuard guardrail. You can try `pip install 'any-guardrail[gliner]'`"
            raise ImportError(msg) from MISSING_PACKAGES_ERROR
        self.model_id = default(model_id, self.SUPPORTED_MODELS)
        self.threshold = threshold
        self.model = GLiNER2.from_pretrained(self.model_id)

    def validate(self, input_text: str, **kwargs: Any) -> GuardrailOutput:
        """Classify ``input_text`` across the safety/toxicity/jailbreak/refusal schema."""
        del kwargs
        start = time.perf_counter()
        result: dict[str, Any] = self.model.classify_text(input_text, GLIGUARD_SCHEMA, threshold=self.threshold)

        prompt_safety = result.get("prompt_safety")
        toxicity = _as_label_list(result.get("prompt_toxicity"))
        jailbreak = [label for label in _as_label_list(result.get("jailbreak_detection")) if label != "benign"]
        refusal = result.get("response_refusal")

        categories = [CategoryResult(name="prompt_safety", description=str(prompt_safety), triggered=prompt_safety == "unsafe")]
        categories += [CategoryResult(name=label, triggered=True) for label in toxicity if label != "benign"]
        categories += [CategoryResult(name=label, triggered=True) for label in jailbreak]

        unsafe = prompt_safety == "unsafe" or bool(jailbreak)
        out = GuardrailOutput(
            valid=not unsafe,
            categories=categories,
            extra={"prompt_safety": prompt_safety, "response_refusal": refusal, "raw": result},
        )
        out.usage = GuardrailUsage(model_id=self.model_id, latency_ms=(time.perf_counter() - start) * 1000.0)
        return out


def _as_label_list(value: Any) -> list[str]:
    """Normalize a gliner2 task result (single label or list) into a list of label strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]
