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

# Constructor params that are execution plumbing / credentials, not user-facing config knobs.
_SKIP_CREATE = frozenset({"self", "cls", "provider", "api_key"})
# Content-registry-backed enum sources, keyed by parameter name.
_CONTENT_CHOICES = {"criteria": list_criteria, "policy": list_policies, "rubric": list_rubrics}


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


def build_payload() -> dict[str, list[dict[str, Any]]]:
    """Build the ``{guardrail_name: [param_spec, ...]}`` mapping (create params, then validate)."""
    payload: dict[str, list[dict[str, Any]]] = {}
    for gname in GuardrailName:
        cls = AnyGuardrail._get_guardrail_class(gname)
        payload[gname.value] = _create_params(cls, gname.value) + _validate_params(cls, gname.value)
    return payload


def render(payload: dict[str, list[dict[str, Any]]]) -> str:
    """Render the committed ``_parameter_data.py`` (payload embedded as a parsed JSON string)."""
    body = json.dumps(payload, indent=2, sort_keys=True)
    return (
        '"""Generated parameter data for the import-free parameter registry (issue #206).\n\n'
        "Auto-generated by ``scripts/generate_parameter_data.py`` from guardrail signatures +\n"
        "docstrings. Do not edit by hand; run ``python scripts/generate_parameter_data.py`` to\n"
        "regenerate. The payload is embedded as a JSON string so this module stays trivially stable.\n"
        '"""\n\n'
        "import json\n"
        "from typing import Any\n\n"
        f'_PARAMETER_DATA_JSON = r"""\n{body}\n"""\n\n'
        "PARAMETER_DATA: dict[str, list[dict[str, Any]]] = json.loads(_PARAMETER_DATA_JSON)\n"
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
