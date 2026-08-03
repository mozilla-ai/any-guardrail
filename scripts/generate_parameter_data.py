"""Generate the stdlib-only parameter-data leaf from guardrail signatures + docstrings (#206).

Introspects every guardrail's ``__init__`` and ``validate`` signatures (plus docstrings) and
writes ``src/any_guardrail/_parameter_data.py`` — a dependency-free data module consumed by the
import-free ``any_guardrail.parameter_registry``. Each parameter is typed (string / integer /
number / boolean / enum / json), marked required-or-not, given its default and (for enums) its
choices, and described from the docstring where available. ``enum`` choices are enriched from the
existing sources: ``SUPPORTED_MODELS`` for ``model_id``, the content registry for
``criteria`` / ``policy`` / ``rubric``, and the prompt registry for ``prompt_version``.

The payload is embedded as a JSON string parsed at import, so the generated module is trivially
stable (byte-for-byte reproducible) and ``--check`` in pre-commit fails on drift.

Usage:
    python scripts/generate_parameter_data.py            # write src/any_guardrail/_parameter_data.py
    python scripts/generate_parameter_data.py --check    # exit non-zero if the committed file is stale
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import types as _types
import typing
from pathlib import Path
from typing import Any

_SCRIPTS = Path(__file__).parent
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_SCRIPTS.parent / "src"))

from generate_api_docs import _parse_args_section  # noqa: E402  (path set up above)

from any_guardrail.api import AnyGuardrail  # noqa: E402
from any_guardrail.base import Guardrail, GuardrailName  # noqa: E402
from any_guardrail.content_registry import list_criteria, list_policies, list_rubrics  # noqa: E402
from any_guardrail.prompt_registry import list_prompt_versions  # noqa: E402

DEFAULT_OUT = _SCRIPTS.parent / "src" / "any_guardrail" / "_parameter_data.py"

# Constructor params that are pure execution plumbing (a live backend object), not config knobs.
# NB: credentials (``api_key`` etc.) are deliberately NOT skipped — they surface with ``secret=True``
# (see ``_SECRET_PARAMS``) so a config UI can render them, mask them, and know they must not be logged.
_SKIP_CREATE = frozenset({"self", "cls", "provider"})
# Content-registry-backed enum sources, keyed by parameter name.
_CONTENT_CHOICES = {"criteria": list_criteria, "policy": list_policies, "rubric": list_rubrics}

# --- Runtime-requirement declarations (issue #206 follow-up) ----------------------------------
# These capture what the *signature* cannot: a value can be mandatory at runtime yet optional as an
# argument, because it falls back to an environment variable or is one of several interchangeable
# parameters. Hand-maintained (env fallbacks live in function bodies / docstrings, not signatures)
# and cross-checked against the guardrail source by tests/unit/test_parameters.py so they cannot
# silently drift. Keyed by snake_case guardrail (``GuardrailName`` value).

# {guardrail: {parameter: ENV_VAR}} — the env var that back-fills the argument when it is not passed.
_ENV_VAR_FALLBACKS: dict[str, dict[str, str]] = {
    "alinia": {"api_key": "ALINIA_API_KEY", "endpoint": "ALINIA_ENDPOINT"},
    "azure_content_safety": {"api_key": "CONTENT_SAFETY_KEY", "endpoint": "CONTENT_SAFETY_ENDPOINT"},
    "azure_prompt_shields": {"api_key": "CONTENT_SAFETY_KEY", "endpoint": "CONTENT_SAFETY_ENDPOINT"},
    "lakera_guard": {"api_key": "LAKERA_API_KEY"},
    "openai_moderation": {"api_key": "OPENAI_API_KEY"},
    "patronus": {"api_key": "PATRONUS_API_KEY"},
    "watsonx_guardian": {
        "api_key": "WATSONX_APIKEY",
        "url": "WATSONX_URL",
        "project_id": "WATSONX_PROJECT_ID",
        "space_id": "WATSONX_SPACE_ID",
    },
}

# {guardrail: {parameter, ...}} — credential parameters (rendered masked; never logged). Includes
# credential-bearing objects (``api_client``, ``boto3_session``) that embed auth, not just strings.
_SECRET_PARAMS: dict[str, frozenset[str]] = {
    "alinia": frozenset({"api_key"}),
    "azure_content_safety": frozenset({"api_key"}),
    "azure_prompt_shields": frozenset({"api_key"}),
    "bedrock_guardrails": frozenset({"aws_access_key_id", "aws_secret_access_key", "boto3_session"}),
    "lakera_guard": frozenset({"api_key"}),
    "openai_moderation": frozenset({"api_key"}),
    "patronus": frozenset({"api_key"}),
    "watsonx_guardian": frozenset({"api_key", "api_client"}),
}

# {guardrail: {parameter, ...}} — params with a signature default that the guardrail nonetheless
# requires a resolved value for (raises if neither the argument nor its env var is set). Members of
# a one-of ``_REQUIREMENT_GROUPS`` entry are recorded there instead, never here — which is why
# ``watsonx_guardian`` is absent: its api_key/url/project_id/space_id are all group members (each
# group also satisfiable by the ``api_client`` escape hatch), so no single one raises unconditionally.
_EFFECTIVELY_REQUIRED: dict[str, frozenset[str]] = {
    "alinia": frozenset({"api_key", "endpoint"}),
    "azure_content_safety": frozenset({"api_key", "endpoint"}),
    "azure_prompt_shields": frozenset({"api_key", "endpoint"}),
    "lakera_guard": frozenset({"api_key"}),
    "openai_moderation": frozenset({"api_key"}),
    "patronus": frozenset({"api_key"}),
}

# {guardrail: [{"description", "parameters": [...], "env_vars": [...]}]} — at-least-one-of groups.
# Only genuine "at least one of these interchangeable params (or their env fallback) must resolve"
# constraints go here. Deliberately EXCLUDED after review:
#   * flowjudge (metric vs. the full name/criteria/rubric/required_inputs/required_output bundle) —
#     an "A or ALL of B" bundle, not "at least one of"; its param descriptions state the rule.
#   * azure_prompt_shields (user_prompt vs. documents at validate time) — user_prompt is the primary
#     input, not a configurable schema parameter, so it never appears in get_parameter_schema.
# watsonx folds its ``api_client`` escape hatch into each credential group as an alternative member,
# so supplying api_client satisfies all three exactly as the runtime ``if api_client is None`` guard does.
_REQUIREMENT_GROUPS: dict[str, list[dict[str, Any]]] = {
    "watsonx_guardian": [
        {
            "description": (
                "An IBM Cloud IAM API key is required: pass ``api_key`` (or set ``WATSONX_APIKEY``), "
                "or supply a pre-built ``api_client``."
            ),
            "parameters": ["api_key", "api_client"],
            "env_vars": ["WATSONX_APIKEY"],
        },
        {
            "description": (
                "A watsonx.ai region URL is required: pass ``url`` (or set ``WATSONX_URL``), "
                "or supply a pre-built ``api_client``."
            ),
            "parameters": ["url", "api_client"],
            "env_vars": ["WATSONX_URL"],
        },
        {
            "description": (
                "A project or space is required: pass ``project_id`` or ``space_id`` (or set "
                "``WATSONX_PROJECT_ID`` / ``WATSONX_SPACE_ID``), or supply a pre-built ``api_client``."
            ),
            "parameters": ["project_id", "space_id", "api_client"],
            "env_vars": ["WATSONX_PROJECT_ID", "WATSONX_SPACE_ID"],
        },
    ],
}


def _classify(annotation: Any) -> tuple[str, tuple[str, ...] | None]:
    """Map a signature annotation (a real type object) to a ``(ParameterType, choices)`` pair."""
    if annotation is inspect.Parameter.empty:
        return "json", None
    origin = typing.get_origin(annotation)
    if origin is typing.Literal:
        return "enum", tuple(str(arg) for arg in typing.get_args(annotation))
    if origin is typing.Union or origin is _types.UnionType:
        non_none = [arg for arg in typing.get_args(annotation) if arg is not type(None)]
        if len(non_none) == 1:
            return _classify(non_none[0])
        return "json", None  # a genuine multi-type union (e.g. str | dict) -> JSON editor
    if origin is not None:
        return "json", None  # a parameterized generic: list[...], dict[...], etc.
    if annotation is bool:
        return "boolean", None
    if annotation is int:
        return "integer", None
    if annotation is float:
        return "number", None
    if annotation is str:
        return "string", None
    return "json", None  # any other class (e.g. PromptTemplate) -> JSON editor


def _enrich_choices(
    name: str, param_name: str, base_type: str, cls: type[Guardrail]
) -> tuple[str, tuple[str, ...] | None]:
    """Override ``(type, choices)`` for parameters whose choices come from an existing registry."""
    gname = GuardrailName(name)
    if param_name == "model_id" and getattr(cls, "SUPPORTED_MODELS", None):
        return "enum", tuple(cls.SUPPORTED_MODELS)
    if param_name == "prompt_version":
        versions = list_prompt_versions(gname)
        if versions:
            return "enum", tuple(versions)
    if param_name in _CONTENT_CHOICES and base_type in ("string", "enum"):
        keys = _CONTENT_CHOICES[param_name](gname)
        if keys:
            return "enum", tuple(keys)
    return base_type, None


def _json_default(param: inspect.Parameter) -> Any:
    """Return the JSON-native default for a parameter (None for required or non-scalar defaults)."""
    default = param.default
    if default is inspect.Parameter.empty or default is None:
        return None
    if isinstance(default, bool | str):
        return default
    if isinstance(default, int | float):
        return default
    return None  # a non-JSON default (object/mutable); its shape is captured by ``type``


def _spec(
    name: str, param: inspect.Parameter, stage: str, cls: type[Guardrail], docs: dict[str, str]
) -> dict[str, Any]:
    base_type, choices = _classify(param.annotation)
    ptype, enriched = _enrich_choices(name, param.name, base_type, cls)
    if enriched is not None:
        choices = enriched
    return {
        "name": param.name,
        "stage": stage,
        "type": ptype,
        "required": param.default is inspect.Parameter.empty,
        "default": _json_default(param),
        "effectively_required": param.name in _EFFECTIVELY_REQUIRED.get(name, frozenset()),
        "env_var": _ENV_VAR_FALLBACKS.get(name, {}).get(param.name),
        "secret": param.name in _SECRET_PARAMS.get(name, frozenset()),
        "choices": list(choices) if choices is not None else None,
        "description": docs.get(param.name),
    }


def _validate_source(cls: type[Guardrail]) -> Any:
    """Return the callable whose params define validate() inputs (mirrors tests/unit/test_metadata.py)."""
    if "validate" in cls.__dict__:
        return cls.validate
    return cls._pre_processing  # type: ignore[attr-defined]


def _create_params(cls: type[Guardrail], name: str) -> list[dict[str, Any]]:
    init = cls.__init__
    docs = {**_parse_args_section(cls.__doc__), **_parse_args_section(inspect.getdoc(init))}
    params = list(inspect.signature(init).parameters.values())
    return [
        _spec(name, param, "create", cls, docs)
        for param in params
        if param.name not in _SKIP_CREATE
        and param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]


def _validate_params(cls: type[Guardrail], name: str) -> list[dict[str, Any]]:
    source = _validate_source(cls)
    docs = _parse_args_section(inspect.getdoc(source))
    params = [
        param
        for param in inspect.signature(source).parameters.values()
        if param.name not in ("self", "cls")
        and param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]
    return [_spec(name, param, "validate", cls, docs) for param in params[1:]]  # skip the primary input


def build_payload() -> dict[str, Any]:
    """Build ``{"parameters": {name: [spec, ...]}, "requirement_groups": {name: [group, ...]}}``.

    ``parameters`` lists each guardrail's create params then validate params; ``requirement_groups``
    carries the hand-declared one-of constraints, only for the guardrails that have them.
    """
    parameters: dict[str, list[dict[str, Any]]] = {}
    for gname in GuardrailName:
        cls = AnyGuardrail._get_guardrail_class(gname)
        parameters[gname.value] = _create_params(cls, gname.value) + _validate_params(cls, gname.value)
    requirement_groups = {name: groups for name, groups in _REQUIREMENT_GROUPS.items() if groups}
    return {"parameters": parameters, "requirement_groups": requirement_groups}


def render(payload: dict[str, Any]) -> str:
    """Render the committed ``_parameter_data.py`` (payload embedded as parsed JSON strings)."""
    params_body = json.dumps(payload["parameters"], indent=2, sort_keys=True)
    groups_body = json.dumps(payload["requirement_groups"], indent=2, sort_keys=True)
    return (
        '"""Generated parameter data for the import-free parameter registry (issue #206).\n\n'
        "Auto-generated by ``scripts/generate_parameter_data.py`` from guardrail signatures +\n"
        "docstrings. Do not edit by hand; run ``python scripts/generate_parameter_data.py`` to\n"
        "regenerate. The payloads are embedded as JSON strings so this module stays trivially stable.\n"
        '"""\n\n'
        "import json\n"
        "from typing import Any\n\n"
        f'_PARAMETER_DATA_JSON = r"""\n{params_body}\n"""\n\n'
        f'_REQUIREMENT_GROUPS_JSON = r"""\n{groups_body}\n"""\n\n'
        "PARAMETER_DATA: dict[str, list[dict[str, Any]]] = json.loads(_PARAMETER_DATA_JSON)\n\n"
        "REQUIREMENT_GROUPS: dict[str, list[dict[str, Any]]] = json.loads(_REQUIREMENT_GROUPS_JSON)\n"
    )


def main() -> int:
    """Generate or check the committed parameter-data leaf."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"Output path (default: {DEFAULT_OUT})")
    parser.add_argument(
        "--check", action="store_true", help="Exit non-zero (without writing) if the committed file is stale."
    )
    args = parser.parse_args()

    content = render(build_payload())

    if args.check:
        if not args.out.exists() or args.out.read_text(encoding="utf-8") != content:
            print(
                f"{args.out} is out of date. Run `python scripts/generate_parameter_data.py` and commit the result.",
                file=sys.stderr,
            )
            return 1
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(content, encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
