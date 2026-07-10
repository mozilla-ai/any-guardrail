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
