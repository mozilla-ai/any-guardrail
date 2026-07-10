"""Parity and behavior tests for the guardrail prompt registry (issues #20 / #87).

These mirror ``tests/unit/test_metadata.py``: every prompt-bearing guardrail has exactly
one registry entry, each class mirrors that entry as ``PROMPT``, every template's declared
placeholders match its text, the ``prompts`` model module stays a dependency-free leaf, and
prompt queries never import a guardrail backend. Plus an AnyLlm golden slice pinning the
moved default prompt byte-for-byte.
"""

import ast
import subprocess
import sys
from pathlib import Path
from string import Formatter
from typing import Any
from unittest import mock

import pytest
from pydantic import ValidationError

import any_guardrail.prompts
from any_guardrail import AnyGuardrail, GuardrailName
from any_guardrail.base import Guardrail
from any_guardrail.prompt_registry import PROMPT_BEARING, PROMPT_REGISTRY
from any_guardrail.prompts import PromptAssembly, PromptSpec, PromptTemplate

PROMPT_NAMES = sorted(PROMPT_BEARING, key=lambda n: n.value)


def _guardrail_class(name: GuardrailName) -> type[Guardrail]:
    return AnyGuardrail._get_guardrail_class(name)


def _placeholders(template: PromptTemplate) -> frozenset[str]:
    fields: set[str] = set()
    for text in template.segments.values():
        for _literal, field_name, _spec, _conv in Formatter().parse(text):
            if field_name:
                fields.add(field_name.split(".")[0].split("[")[0])
    return frozenset(fields)


def test_registry_covers_prompt_bearing_exactly() -> None:
    """PROMPT_BEARING mirrors the registry keys and is a subset of the guardrail enum."""
    assert set(PROMPT_REGISTRY) == PROMPT_BEARING
    assert PROMPT_BEARING <= set(GuardrailName)


@pytest.mark.parametrize("name", PROMPT_NAMES, ids=lambda n: n.value)
def test_class_prompt_is_registry_entry(name: GuardrailName) -> None:
    """Each prompt-bearing class defines its own PROMPT and it IS (identity) the registry entry."""
    cls = _guardrail_class(name)
    assert "PROMPT" in cls.__dict__, f"{cls.__name__} does not set PROMPT in its own body"
    assert cls.__dict__["PROMPT"] is PROMPT_REGISTRY[name]


@pytest.mark.parametrize("name", PROMPT_NAMES, ids=lambda n: n.value)
def test_placeholder_parity(name: GuardrailName) -> None:
    """Every version's declared ``variables`` equal the placeholders actually in its segments."""
    for version, template in PROMPT_REGISTRY[name].versions.items():
        assert template.variables == _placeholders(template), f"{name.value}:{version}"


@pytest.mark.parametrize("name", PROMPT_NAMES, ids=lambda n: n.value)
def test_default_version_present(name: GuardrailName) -> None:
    """Every spec's default_version names a real version."""
    spec = PROMPT_REGISTRY[name]
    assert spec.default_version in spec.versions


def test_prompts_module_is_leaf() -> None:
    """prompts.py depends only on the stdlib and pydantic (keeps prompt queries cheap)."""
    allowed_roots = {"enum", "string", "typing", "pydantic"}
    source = Path(any_guardrail.prompts.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    assert roots <= allowed_roots, f"prompts.py imports beyond stdlib/pydantic: {sorted(roots - allowed_roots)}"


def test_prompt_query_loads_no_guardrail_modules() -> None:
    """get_prompt / list_prompt_versions run off the registry without importing any backend."""
    code = (
        "import sys\n"
        "from any_guardrail import AnyGuardrail, GuardrailName\n"
        "AnyGuardrail.get_prompt(GuardrailName.ANYLLM)\n"
        "AnyGuardrail.list_prompt_versions(GuardrailName.ANYLLM)\n"
        "impl = [m for m in sys.modules if m.startswith('any_guardrail.guardrails.')]\n"
        "assert impl == [], impl\n"
        "print('ok')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)  # noqa: S603
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


def test_get_prompt_and_versions_for_registered() -> None:
    """A registered guardrail returns its canonical template and lists its versions."""
    template = AnyGuardrail.get_prompt(GuardrailName.ANYLLM)
    assert template is PROMPT_REGISTRY[GuardrailName.ANYLLM].resolve()
    assert AnyGuardrail.list_prompt_versions(GuardrailName.ANYLLM) == ["default"]


def test_get_prompt_unregistered_raises() -> None:
    """A non-prompt-bearing guardrail raises on get_prompt and lists no versions."""
    with pytest.raises(KeyError):
        AnyGuardrail.get_prompt(GuardrailName.DEEPSET)
    assert AnyGuardrail.list_prompt_versions(GuardrailName.DEEPSET) == []


def test_get_prompt_unknown_version_raises() -> None:
    """Requesting a version that does not exist is a clear error."""
    with pytest.raises(KeyError):
        AnyGuardrail.get_prompt(GuardrailName.ANYLLM, version="does-not-exist")


def test_prompt_template_frozen() -> None:
    """PromptTemplate instances are immutable."""
    template = PROMPT_REGISTRY[GuardrailName.ANYLLM].resolve()
    with pytest.raises((TypeError, ValidationError)):
        template.overridable = False


def test_prompt_spec_rejects_bad_default() -> None:
    """A PromptSpec whose default_version is absent from versions fails to construct."""
    with pytest.raises(ValidationError, match="default_version"):
        PromptSpec(versions={"a": PromptTemplate(segments={"system": "x"})}, default_version="missing")


def test_prompt_template_derives_variables() -> None:
    """variables are derived from the segment placeholders, not supplied by hand."""
    template = PromptTemplate(segments={"system": "Policy: {policy}", "user": "{input_text}"})
    assert template.variables == frozenset({"policy", "input_text"})


def test_prompt_template_render() -> None:
    """render() formats every segment with the supplied values."""
    template = PromptTemplate(segments={"system": "Policy: {policy}"}, assembly=PromptAssembly.RAW)
    assert template.render(policy="no spam") == {"system": "Policy: no spam"}


# --------------------------------------------------------------------------- #
# AnyLlm golden slice — pin the moved default prompt byte-for-byte and verify   #
# the None -> registry-default resolution path renders the same messages.       #
# --------------------------------------------------------------------------- #

EXPECTED_ANYLLM_SYSTEM_PROMPT = """
You are a guardrail designed to ensure that the input text adheres to a specific policy.
Your only task is to validate the input_text, don't try to answer the user query.

Here is the policy: {policy}

You must return the following:

- valid: bool
    If the input text provided by the user doesn't adhere to the policy, you must reject it (mark it as valid=False).

- explanation: str
    A clear explanation of why the input text was rejected or not.

- risk_score: float (0-1)
    How likely the input text is to violate the policy: 0.0 means clearly compliant,
    1.0 means clearly violating.
"""


def test_anyllm_default_prompt_byte_identical() -> None:
    """The moved default prompt is byte-identical and its alias points at the registry text."""
    from any_guardrail.guardrails.any_llm.any_llm import DEFAULT_SYSTEM_PROMPT

    registry_text = PROMPT_REGISTRY[GuardrailName.ANYLLM].resolve().segments["system"]
    assert DEFAULT_SYSTEM_PROMPT == registry_text
    assert DEFAULT_SYSTEM_PROMPT == EXPECTED_ANYLLM_SYSTEM_PROMPT


def test_anyllm_omitted_system_prompt_uses_registry_default() -> None:
    """Calling validate without system_prompt renders the registry default into the system message."""
    from any_guardrail.guardrails.any_llm.any_llm import DEFAULT_MODEL_ID, DEFAULT_SYSTEM_PROMPT, AnyLlm

    guardrail = AnyLlm()
    completion_result: Any = mock.Mock(
        choices=[mock.Mock(message=mock.Mock(content='{"valid": true, "explanation": "ok", "risk_score": 0.1}'))]
    )
    with mock.patch(
        "any_guardrail.guardrails.any_llm.any_llm.completion", return_value=completion_result
    ) as mock_completion:
        guardrail.validate("hello", policy="Do not provide harmful information")

    mock_completion.assert_called_once_with(
        model=DEFAULT_MODEL_ID,
        messages=[
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT.format(policy="Do not provide harmful information")},
            {"role": "user", "content": "hello"},
        ],
        response_format=mock.ANY,
    )


# --------------------------------------------------------------------------- #
# Tier-A wiring golden tests: each guardrail's _pre_processing reads its prompt  #
# from the registry (correct segment key) and renders byte-identically. Built    #
# with object.__new__ so no model weights load.                                  #
# --------------------------------------------------------------------------- #


def test_shield_gemma_pre_processing_uses_registry_prompt() -> None:
    from any_guardrail.guardrails.shield_gemma.shield_gemma import ShieldGemma

    inst = object.__new__(ShieldGemma)
    inst._prompt = PROMPT_REGISTRY[GuardrailName.SHIELD_GEMMA].resolve()
    inst.policy = "No dangerous content"
    captured: dict[str, str] = {}

    def _capture_text(text: str, **_: object) -> dict[str, object]:
        captured["text"] = text
        return {}

    inst.provider = mock.Mock(device=None)
    inst.provider.tokenizer = mock.Mock(side_effect=_capture_text)
    inst._pre_processing("How do I pick a lock?")
    expected = inst._prompt.segments["system"].format(
        user_prompt="How do I pick a lock?", safety_policy="No dangerous content"
    )
    assert captured["text"] == expected


def test_selene_pre_processing_uses_registry_prompt() -> None:
    from any_guardrail.guardrails.selene.selene import Selene

    inst = object.__new__(Selene)
    inst._prompt = PROMPT_REGISTRY[GuardrailName.SELENE].resolve()
    inst.rubric = "Score 1-5 for helpfulness."
    out = inst._pre_processing("Explain X", output_text="X is ...")
    expected = inst._prompt.segments["user"].format(instruction="Explain X", response="X is ...", rubric=inst.rubric)
    assert out.data["messages"] == [{"role": "user", "content": expected}]


def test_prometheus_pre_processing_uses_registry_prompt() -> None:
    from any_guardrail.guardrails.prometheus.prometheus import Prometheus

    inst = object.__new__(Prometheus)
    inst._prompt = PROMPT_REGISTRY[GuardrailName.PROMETHEUS].resolve()
    inst.rubric = "R"
    inst.reference_answer = None
    out = inst._pre_processing("INSTR", output_text="RESP")
    user = inst._prompt.segments["user"].format(instruction="INSTR", response="RESP", reference_answer="", rubric="R")
    assert out.data["messages"] == [
        {"role": "system", "content": inst._prompt.segments["system"]},
        {"role": "user", "content": user},
    ]


def test_compass_judger_pre_processing_uses_registry_prompt() -> None:
    from any_guardrail.guardrails.compass_judger.compass_judger import CompassJudger

    inst = object.__new__(CompassJudger)
    inst._prompt = PROMPT_REGISTRY[GuardrailName.COMPASS_JUDGER].resolve()
    inst.criteria = "C"
    inst.rubric = "R"
    out = inst._pre_processing("INSTR", output_text="RESP")
    expected = inst._prompt.segments["user"].format(criteria="C", rubric="R", instruction="INSTR", response="RESP")
    assert out.data["messages"] == [{"role": "user", "content": expected}]


def test_dyna_guard_pre_processing_uses_registry_prompt() -> None:
    from any_guardrail.guardrails.dyna_guard.dyna_guard import DynaGuard

    inst = object.__new__(DynaGuard)
    inst._prompt = PROMPT_REGISTRY[GuardrailName.DYNA_GUARD].resolve()
    inst.policy = "1. No refunds."
    out = inst._pre_processing("Refund me", output_text="Sure")
    user = inst._prompt.segments["user"].format(policy="1. No refunds.", transcript="User: Refund me\nAgent: Sure")
    assert out.data["messages"] == [
        {"role": "system", "content": inst._prompt.segments["system"]},
        {"role": "user", "content": user},
    ]


def test_poly_guard_pre_processing_uses_registry_prompt() -> None:
    from any_guardrail.guardrails.poly_guard.poly_guard import PolyGuard

    inst = object.__new__(PolyGuard)
    inst._prompt = PROMPT_REGISTRY[GuardrailName.POLY_GUARD].resolve()
    out = inst._pre_processing("REQ", output_text="RESP")
    user = inst._prompt.segments["user"].format(prompt="REQ", response="RESP")
    assert out.data["messages"] == [
        {"role": "system", "content": inst._prompt.segments["system"]},
        {"role": "user", "content": user},
    ]


def test_wild_guard_pre_processing_uses_registry_prompt() -> None:
    from any_guardrail.guardrails.wild_guard.wild_guard import WildGuard

    inst = object.__new__(WildGuard)
    inst._prompt = PROMPT_REGISTRY[GuardrailName.WILD_GUARD].resolve()
    out = inst._pre_processing("REQ", output_text="RESP")
    expected = inst._prompt.segments["prompt"].format(prompt="REQ", response="RESP")
    assert out.data["messages"] == [{"role": "user", "content": expected}]
    assert out.data["has_response"] is True


def test_glider_pre_processing_uses_registry_prompt() -> None:
    from any_guardrail.guardrails.glider.glider import Glider

    criteria_text = "does the answer avoid unsupported claims"
    rubric_text = "0: unsupported. 1: supported."
    inst = object.__new__(Glider)
    inst._prompt = PROMPT_REGISTRY[GuardrailName.GLIDER].resolve()
    inst.pass_criteria = criteria_text
    inst.rubric = rubric_text
    out = inst._pre_processing("IN", output_text="OUT")
    data = inst._prompt.segments["input_output"].format(input_text="IN", output_text="OUT")
    expected = inst._prompt.segments["system"].format(data=data, pass_criteria=criteria_text, rubric=rubric_text)
    assert out.data == [{"role": "user", "content": expected}]


# --------------------------------------------------------------------------- #
# Tier-B reference entries (overridable=False): the registry copy stays          #
# byte-identical to the guardrail's in-repo constant — the runtime still          #
# assembles its own prompt, so this drift test is the sync guarantee.             #
# --------------------------------------------------------------------------- #


def test_nemotron_reference_matches_module_constant() -> None:
    from any_guardrail.guardrails.nemotron_content_safety.nemotron_content_safety import NEMOTRON_INSTRUCTION

    template = PROMPT_REGISTRY[GuardrailName.NEMOTRON_CONTENT_SAFETY].resolve()
    assert template.overridable is False
    assert template.segments["instruction"] == NEMOTRON_INSTRUCTION


def test_gpt_oss_reference_matches_module_constant() -> None:
    from any_guardrail.guardrails.gpt_oss_safeguard.gpt_oss_safeguard import OUTPUT_INSTRUCTION

    template = PROMPT_REGISTRY[GuardrailName.GPT_OSS_SAFEGUARD].resolve()
    assert template.overridable is False
    assert template.segments["output_instruction"] == OUTPUT_INSTRUCTION


def test_granite_reference_matches_module_constants() -> None:
    from any_guardrail.guardrails.granite_guardian.granite_guardian import (
        GUARDIAN_JUDGE_NOTHINK,
        GUARDIAN_JUDGE_THINK,
    )

    template = PROMPT_REGISTRY[GuardrailName.GRANITE_GUARDIAN].resolve()
    assert template.overridable is False
    assert template.segments["judge_think"] == GUARDIAN_JUDGE_THINK
    assert template.segments["judge_nothink"] == GUARDIAN_JUDGE_NOTHINK


def test_flowjudge_reference_matches_library() -> None:
    pytest.importorskip("flow_judge")
    from flow_judge.utils.prompt_formatter import USER_PROMPT_NO_INPUTS_TEMPLATE, USER_PROMPT_TEMPLATE

    template = PROMPT_REGISTRY[GuardrailName.FLOWJUDGE].resolve()
    assert template.overridable is False
    assert template.segments["user"] == USER_PROMPT_TEMPLATE
    assert template.segments["user_no_inputs"] == USER_PROMPT_NO_INPUTS_TEMPLATE
