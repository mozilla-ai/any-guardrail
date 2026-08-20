"""Hand-authored value shapes for the guardrails' JSON parameters (issue #206 follow-up).

A parameter annotated ``list[dict]``, ``dict[str, str]``, or ``str | dict`` classifies as
:attr:`~any_guardrail.parameters.ParameterType.JSON`, which tells a config UI only that the value
nests. That is why configuring such a parameter today means hand-writing JSON. This module supplies
the missing structure — see :class:`~any_guardrail.parameters.ParameterShape` — so those parameters
can be rendered as real controls instead.

None of it is inferable from a signature or a docstring, so unlike ``_parameter_data`` this table is
written by hand rather than generated: the shapes and the option/field vocabularies are harvested
from each vendor's own documentation. Provenance is recorded per guardrail below. As with
``_authored_content_data`` and ``_authored_prompt_data``, those sources are not pip-installed, so
there is no live drift test; ``tests/unit/test_parameters.py`` pins the table against the registry
(every JSON parameter is covered, every entry names a parameter that still exists and is still
JSON, and every entry is internally consistent), but it cannot pin a vendor's vocabulary.

Which means: **a vocabulary here is a suggestion, never a constraint.** Providers add evaluators,
detections, and criteria without announcing them, and several are account-specific. Consumers must
keep a way to supply a value this table has never heard of — ``suggestions`` rather than ``choices``
on the open fields, and a raw-JSON fallback for whole values these shapes cannot express. Do not
turn this table into validation.

Coverage is deliberately incremental. A shape is added only once the vendor's own documentation
has been read closely enough to describe the value honestly; an undescribed JSON parameter keeps
``shape=None`` and behaves exactly as it did before this module existed. Guessing a shape is worse
than declaring none, because a wrong control teaches users a vocabulary the provider does not have.

Stdlib-only leaf: imported by ``parameter_registry``, imports nothing itself. Keys are
``{guardrail: {parameter: shape_fields}}``, where ``shape_fields`` is the subset of
:class:`~any_guardrail.parameters.ParameterSpec` fields this layer contributes.
"""

from typing import Any

# Alinia: https://docs.alinia.ai/ — four top-level detection policies. The nested per-policy form
# (``{"safety": {"toxicity": 0.8}}``) keys sub-detections, not a single threshold, and Alinia
# publishes no enumeration of those sub-detections, so only the policy on/off form is described
# here. A nested configuration therefore has no declared shape and belongs in a raw-JSON fallback.
_ALINIA: dict[str, dict[str, Any]] = {
    "detection_config": {
        "shape": "option_map",
        "options": (
            {
                "value": "security",
                "label": "Security",
                "description": "Prompt injection, jailbreak, and data-exfiltration attempts.",
            },
            {
                "value": "safety",
                "label": "Safety",
                "description": "Harmful, toxic, and otherwise unsafe content.",
            },
            {
                "value": "compliance",
                "label": "Compliance",
                "description": "Policy and regulatory violations.",
            },
            {
                "value": "hallucination",
                "label": "Hallucination",
                "description": "Claims unsupported by the provided context. Needs context documents.",
            },
        ),
        "scalar_alternative": {
            "key": "detection_config_id",
            "label": "Detection configuration ID",
            "description": "Use a detection configuration already registered with Alinia instead of "
            "choosing detections here.",
        },
    },
    "metadata": {"shape": "string_map"},
    "blocked_response": {"shape": "string_map"},
    "context_documents": {"shape": "string_list"},
}

# Azure AI Content Safety: blocklists are referenced by name, created out-of-band in the Azure
# portal. https://learn.microsoft.com/azure/ai-services/content-safety/
_AZURE_CONTENT_SAFETY: dict[str, dict[str, Any]] = {
    "blocklist_names": {"shape": "string_list"},
}

# Bedrock: a live, pre-configured ``boto3.Session``. Not expressible as configuration.
_BEDROCK_GUARDRAILS: dict[str, dict[str, Any]] = {
    "boto3_session": {"shape": "opaque"},
}

# Lakera Guard: https://platform.lakera.ai/docs — free-form observability metadata.
_LAKERA_GUARD: dict[str, dict[str, Any]] = {
    "metadata": {"shape": "string_map"},
}

# Patronus: evaluator families and managed criteria aliases from
# https://docs.patronus.ai/docs/evaluators/reference_guide. Criteria are partly account-specific
# (custom judge configurations live under the caller's own account and are listed nowhere public),
# so both columns stay open. ``explain_strategy`` is left open for the same reason.
_PATRONUS: dict[str, dict[str, Any]] = {
    "evaluators": {
        "shape": "object_list",
        "item_fields": (
            {
                "key": "evaluator",
                "label": "Evaluator",
                "required": True,
                "suggestions": (
                    "judge",
                    "lynx",
                    "toxicity",
                    "pii",
                    "phi",
                    "answer-relevance",
                    "context-relevance",
                    "context-sufficiency",
                    "hallucination",
                    "glider",
                ),
                "description": "The Patronus evaluator family to run.",
            },
            {
                "key": "criteria",
                "label": "Criteria",
                "suggestions": (
                    "patronus:prompt-injection",
                    "patronus:toxicity",
                    "patronus:pii",
                    "patronus:hallucination",
                ),
                "description": "A managed criteria alias, or the name of a criteria registered with "
                "your Patronus account.",
            },
            {
                "key": "explain_strategy",
                "label": "Explanation",
                "suggestions": ("never", "on-fail", "on-success", "always"),
                "description": "When Patronus should return an explanation with the verdict.",
            },
        ),
        "presets": (
            {
                "label": "Prompt injection",
                "value": {"evaluator": "judge", "criteria": "patronus:prompt-injection"},
                "category": "prompt_injection",
                "description": "Jailbreak and instruction-override attempts in the prompt.",
            },
            {
                "label": "Toxicity",
                "value": {"evaluator": "toxicity", "criteria": "patronus:toxicity"},
                "category": "toxicity",
                "description": "Abusive, hateful, or harassing language.",
            },
            {
                "label": "PII",
                "value": {"evaluator": "pii", "criteria": "patronus:pii"},
                "category": "pii",
                "description": "Personally identifiable information in the text.",
            },
            {
                "label": "Hallucination",
                "value": {"evaluator": "lynx"},
                "category": "hallucination",
                "description": "Claims unsupported by the retrieved context. Needs retrieved "
                "context and the model output.",
            },
            {
                "label": "Answer relevance",
                "value": {"evaluator": "answer-relevance"},
                "category": "off_topic",
                "description": "Whether the response actually answers the prompt. Needs the model output.",
            },
        ),
    },
    "tags": {"shape": "string_map"},
    "retrieved_context": {"shape": "string_list"},
}

# watsonx.ai Guardian: the ``granite_guardian``, ``hap``, and ``pii`` detectors, each taking a
# ``threshold`` in [0, 1]. https://www.ibm.com/products/watsonx-ai
_WATSONX_GUARDIAN: dict[str, dict[str, Any]] = {
    "detectors": {
        "shape": "option_map",
        "options": (
            {
                "value": "granite_guardian",
                "label": "Granite Guardian",
                "description": "General harm detection across the Granite Guardian risk taxonomy.",
                "knob": "threshold",
                "knob_min": 0.0,
                "knob_max": 1.0,
            },
            {
                "value": "hap",
                "label": "Hate, abuse, and profanity",
                "knob": "threshold",
                "knob_min": 0.0,
                "knob_max": 1.0,
            },
            {
                "value": "pii",
                "label": "PII",
                "description": "Personally identifiable information.",
                "knob": "threshold",
                "knob_min": 0.0,
                "knob_max": 1.0,
            },
        ),
    },
    "api_client": {"shape": "opaque"},
}

PARAMETER_SHAPES: dict[str, dict[str, dict[str, Any]]] = {
    "alinia": _ALINIA,
    "azure_content_safety": _AZURE_CONTENT_SAFETY,
    "bedrock_guardrails": _BEDROCK_GUARDRAILS,
    "lakera_guard": _LAKERA_GUARD,
    "patronus": _PATRONUS,
    "watsonx_guardian": _WATSONX_GUARDIAN,
}
