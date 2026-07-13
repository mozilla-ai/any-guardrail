import importlib
import inspect
import re
from collections.abc import Iterable
from typing import Any

import any_guardrail.content_registry as _content
from any_guardrail.base import Guardrail, GuardrailName
from any_guardrail.prompt_registry import get_prompt as _registry_get_prompt
from any_guardrail.prompt_registry import list_prompt_versions as _registry_list_prompt_versions
from any_guardrail.prompts import PromptTemplate
from any_guardrail.providers.base import Provider
from any_guardrail.registry import GUARDRAIL_METADATA
from any_guardrail.taxonomy import (
    BackendType,
    GuardrailCategory,
    GuardrailMetadata,
    GuardrailStage,
    OutputShape,
)

# Metadata dimensions that hold a set of values (matched by any-overlap) vs a
# single scalar value (matched by equality) when filtering / grouping.
_SET_DIMENSIONS = {"category": "categories", "stage": "stages", "output_shape": "output_shapes"}
_SCALAR_DIMENSIONS = {"backend": "backend", "vendor": "vendor"}


class AnyGuardrail:
    """Factory class for creating guardrail instances."""

    @classmethod
    def get_supported_guardrails(cls) -> list[GuardrailName]:
        """List all supported guardrails."""
        return list(GuardrailName)

    @classmethod
    def metadata(cls, guardrail_name: GuardrailName) -> GuardrailMetadata:
        """Return the static taxonomy/capability metadata for a guardrail.

        Args:
            guardrail_name: The guardrail to look up.

        Returns:
            The guardrail's :class:`~any_guardrail.taxonomy.GuardrailMetadata`. This
            reads the import-free registry and does not import the guardrail
            implementation or any model backend.

        """
        return GUARDRAIL_METADATA[guardrail_name]

    @classmethod
    def list_guardrails(
        cls,
        *,
        category: GuardrailCategory | Iterable[GuardrailCategory] | None = None,
        stage: GuardrailStage | Iterable[GuardrailStage] | None = None,
        output_shape: OutputShape | Iterable[OutputShape] | None = None,
        backend: BackendType | None = None,
        requires_api_key: bool | None = None,
        multilingual: bool | None = None,
        multimodal: bool | None = None,
        vendor: str | None = None,
    ) -> list[GuardrailName]:
        """List guardrails whose metadata matches every supplied filter.

        Filters combine with AND across dimensions. Set-valued dimensions
        (``category``, ``stage``, ``output_shape``) accept a single enum member or
        an iterable and match when the guardrail has **any** of the requested
        values. Scalar dimensions match by equality. A ``None`` filter is ignored.
        Runs entirely against the import-free registry — no backends are loaded.

        Args:
            category: Keep guardrails detecting any of these categories.
            stage: Keep guardrails that run at any of these stages.
            output_shape: Keep guardrails producing any of these output shapes.
            backend: Keep guardrails with this backend.
            requires_api_key: Keep guardrails matching this API-key requirement.
            multilingual: Keep guardrails matching this multilingual flag.
            multimodal: Keep guardrails matching this multimodal flag.
            vendor: Keep guardrails from this vendor.

        Returns:
            Matching ``GuardrailName`` members, in ``GuardrailName`` declaration order.

        """

        def as_set(value: Any) -> set[Any] | None:
            if value is None:
                return None
            if isinstance(value, Iterable) and not isinstance(value, str):
                return set(value)
            return {value}

        want_category = as_set(category)
        want_stage = as_set(stage)
        want_output_shape = as_set(output_shape)

        result: list[GuardrailName] = []
        for name in GuardrailName:
            meta = GUARDRAIL_METADATA[name]
            if want_category is not None and meta.categories.isdisjoint(want_category):
                continue
            if want_stage is not None and meta.stages.isdisjoint(want_stage):
                continue
            if want_output_shape is not None and meta.output_shapes.isdisjoint(want_output_shape):
                continue
            if backend is not None and meta.backend != backend:
                continue
            if requires_api_key is not None and meta.requires_api_key != requires_api_key:
                continue
            if multilingual is not None and meta.multilingual != multilingual:
                continue
            if multimodal is not None and meta.multimodal != multimodal:
                continue
            if vendor is not None and meta.vendor != vendor:
                continue
            result.append(name)
        return result

    @classmethod
    def group_by(cls, dimension: str) -> dict[str, list[GuardrailName]]:
        """Group guardrails by one metadata dimension.

        Args:
            dimension: One of ``"category"``, ``"stage"``, ``"output_shape"``,
                ``"backend"``, or ``"vendor"``. For set-valued dimensions a guardrail
                appears under every value it carries.

        Returns:
            A mapping of dimension value (the enum ``value`` string, or the vendor
            name) to the matching ``GuardrailName`` members in declaration order,
            with keys sorted.

        Raises:
            ValueError: If ``dimension`` is not a supported grouping dimension.

        """
        groups: dict[str, list[GuardrailName]] = {}
        if dimension in _SET_DIMENSIONS:
            attr = _SET_DIMENSIONS[dimension]
            for name in GuardrailName:
                for member in getattr(GUARDRAIL_METADATA[name], attr):
                    groups.setdefault(member.value, []).append(name)
        elif dimension in _SCALAR_DIMENSIONS:
            attr = _SCALAR_DIMENSIONS[dimension]
            for name in GuardrailName:
                value = getattr(GUARDRAIL_METADATA[name], attr)
                key = value.value if hasattr(value, "value") else str(value)
                groups.setdefault(key, []).append(name)
        else:
            supported = sorted({*_SET_DIMENSIONS, *_SCALAR_DIMENSIONS})
            msg = f"Unknown grouping dimension {dimension!r}; expected one of {supported}."
            raise ValueError(msg)
        return {key: groups[key] for key in sorted(groups)}

    @classmethod
    def get_prompt(cls, guardrail_name: GuardrailName, version: str | None = None) -> PromptTemplate:
        """Return a guardrail's default (or a named) prompt template.

        Args:
            guardrail_name: The prompt-bearing guardrail to look up.
            version: A specific prompt version key; ``None`` selects the guardrail's default.

        Returns:
            The :class:`~any_guardrail.prompts.PromptTemplate`. Reads the import-free prompt
            registry and does not import the guardrail implementation or any model backend.

        Raises:
            KeyError: If the guardrail has no registered prompt, or ``version`` is unknown.

        """
        return _registry_get_prompt(guardrail_name, version)

    @classmethod
    def list_prompt_versions(cls, guardrail_name: GuardrailName) -> list[str]:
        """Return the registered prompt version keys for a guardrail.

        Returns an empty list for guardrails that have no registered prompt (e.g. encoder
        classifiers). Reads the import-free prompt registry; no backend is loaded.
        """
        return _registry_list_prompt_versions(guardrail_name)

    @classmethod
    def list_policies(cls, guardrail_name: GuardrailName) -> list[str]:
        """List the author-published policy keys registered for a guardrail (empty if none).

        Reads the import-free content registry; no backend is loaded.
        """
        return _content.list_policies(guardrail_name)

    @classmethod
    def get_policy(cls, guardrail_name: GuardrailName, key: str) -> str:
        """Return a ready-to-use author-published policy string for a guardrail.

        Pass the result straight to the guardrail, e.g.
        ``AnyGuardrail.create(GuardrailName.SHIELD_GEMMA, policy=AnyGuardrail.get_policy(...))``.

        Raises:
            KeyError: If the guardrail has no policy with that key.

        """
        return _content.get_policy(guardrail_name, key)

    @classmethod
    def list_rubrics(cls, guardrail_name: GuardrailName) -> list[str]:
        """List the author-published rubric keys registered for a guardrail (empty if none)."""
        return _content.list_rubrics(guardrail_name)

    @classmethod
    def get_rubric(cls, guardrail_name: GuardrailName, key: str) -> str:
        """Return a ready-to-use author-published rubric string for a guardrail.

        Raises:
            KeyError: If the guardrail has no rubric with that key.

        """
        return _content.get_rubric(guardrail_name, key)

    @classmethod
    def list_criteria(cls, guardrail_name: GuardrailName) -> list[str]:
        """List the author-published criteria keys registered for a guardrail (empty if none)."""
        return _content.list_criteria(guardrail_name)

    @classmethod
    def get_criteria(cls, guardrail_name: GuardrailName, key: str) -> str:
        """Return a ready-to-use author-published criterion string for a guardrail.

        Raises:
            KeyError: If the guardrail has no criterion with that key.

        """
        return _content.get_criteria(guardrail_name, key)

    @classmethod
    def get_supported_model(cls, guardrail_name: GuardrailName) -> list[str]:
        """Get the model IDs supported by a specific guardrail."""
        guardrail_class = cls._get_guardrail_class(guardrail_name)
        return guardrail_class.SUPPORTED_MODELS

    @classmethod
    def get_all_supported_models(cls) -> dict[str, list[str]]:
        """Get all model IDs supported by all guardrails."""
        model_ids = {}
        for guardrail_name in cls.get_supported_guardrails():
            model_ids[guardrail_name.value] = cls.get_supported_model(guardrail_name)
        return model_ids

    @classmethod
    def create(
        cls,
        guardrail_name: GuardrailName,
        provider: Provider[Any, Any] | None = None,
        **kwargs: Any,
    ) -> Guardrail:
        """Create a guardrail instance.

        Args:
            guardrail_name: The name of the guardrail to use.
            provider: Optional provider instance to use for model loading and inference.
            **kwargs: Additional keyword arguments to pass to the guardrail constructor.

        Returns:
            A guardrail instance.

        """
        guardrail_class = cls._get_guardrail_class(guardrail_name)
        if provider is not None:
            kwargs["provider"] = provider
        return guardrail_class(**kwargs)

    @classmethod
    def _get_guardrail_class(cls, guardrail_name: GuardrailName) -> type[Guardrail]:
        guardrail_module_name = f"{guardrail_name.value}"
        module_path = f"any_guardrail.guardrails.{guardrail_module_name}.{guardrail_module_name}"

        module = importlib.import_module(module_path)
        parts = re.split(r"[^A-Za-z0-9]+", guardrail_module_name)
        candidate_name = "".join(p.capitalize() for p in parts if p)
        guardrail_class = getattr(module, candidate_name, None)
        if inspect.isclass(guardrail_class) and issubclass(guardrail_class, Guardrail):
            return guardrail_class
        msg = f"Could not resolve guardrail class for '{guardrail_module_name}' in {module.__name__}"
        raise ImportError(msg)
