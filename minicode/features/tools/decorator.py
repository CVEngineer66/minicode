"""``@builtin_tool`` — lowers boilerplate for builtin tool registration.

The ``ToolRegistrar.register`` decorator inspects a Python function's
signature and docstring, produces a JSON Schema for ``ToolSpec.input_schema``,
and wraps the function so ``ToolGraphAdapter`` can call it with the existing
``(arguments: dict, context: ToolContext) -> ToolResult`` contract.

All four permission/execution gates still run — this module only replaces the
declaration boilerplate, not the dispatch path. Parameters named ``context``
(typed as ``ToolContext``) are treated as *injected*: they do not appear in
the schema, and the decorator passes the ``ToolContext`` received from the
adapter.

Typical use::

    tools = ToolRegistrar()

    @tools.register(
        capability=ToolCapability(writes_files=True, concurrency_safe=False),
        permission_policy=PermissionPolicy(kind="write_file", always_require_approval=True),
    )
    def write_file(path: str, content: str, append: bool = False, *, context: ToolContext) -> ToolResult:
        '''Write text to a file.'''
        ...

    registry = ToolRegistry(tools.specs())
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, get_args, get_origin, get_type_hints

from minicode.core.types import (
    PermissionPolicy,
    ToolCapability,
    ToolContext,
    ToolResult,
    ToolSpec,
)
from minicode.features.tools.metadata import enrich_input_schema, enrich_tool_description


_CONTEXT_PARAM = "context"


def _type_to_schema(annotation: Any) -> dict[str, Any]:
    """Map a Python type annotation to a JSON Schema type fragment.

    Unknown / ``Any`` / unannotated -> empty fragment (permissive, matches the
    historical hand-written schemas where a field typed as ``body: {}`` meant
    "any JSON value"). ``Optional[T]`` and ``T | None`` are unwrapped to T;
    optionality is expressed via the ``required`` list instead.
    """
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {}
    origin = get_origin(annotation)
    if origin is not None:
        args = get_args(annotation)
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1 and len(args) == 2:
            annotation = non_none[0]
            origin = get_origin(annotation)
    if annotation is str:
        return {"type": "string"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is bool:
        return {"type": "boolean"}
    if annotation is float:
        return {"type": "number"}
    if annotation is list or origin is list:
        return {"type": "array"}
    if annotation is dict or origin is dict:
        return {"type": "object"}
    return {}


def _build_schema(func: Callable[..., Any]) -> tuple[dict[str, Any], list[str], bool]:
    """Introspect ``func`` and return ``(input_schema, param_names, wants_context)``.

    ``param_names`` lists the schema-visible parameters in declaration order
    so the executor wrapper can copy them from the ``arguments`` dict.
    ``wants_context`` is True iff the function takes a keyword-only
    ``context`` parameter.
    """
    sig = inspect.signature(func)
    try:
        hints = get_type_hints(func)
    except Exception:
        hints = {}
    properties: dict[str, Any] = {}
    required: list[str] = []
    param_names: list[str] = []
    wants_context = False
    for name, param in sig.parameters.items():
        if name == _CONTEXT_PARAM:
            wants_context = True
            continue
        param_names.append(name)
        annotation = hints.get(name, inspect.Parameter.empty)
        properties[name] = _type_to_schema(annotation)
        if param.default is inspect.Parameter.empty:
            required.append(name)
    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema, param_names, wants_context


def _passthrough_validator(arguments: dict[str, Any]) -> dict[str, Any]:
    return dict(arguments or {})


def _make_executor(
    func: Callable[..., ToolResult],
    param_names: list[str],
    wants_context: bool,
) -> Callable[[dict[str, Any], ToolContext], ToolResult]:
    """Build the ``(arguments, context)`` wrapper the adapter calls.

    Only keys present in ``arguments`` are forwarded — missing optional fields
    fall back to the function's own parameter defaults. Extra keys in
    ``arguments`` are silently dropped (models occasionally hallucinate extra
    fields; dropping them is friendlier than raising TypeError).
    """

    def executor(arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        kwargs: dict[str, Any] = {}
        for name in param_names:
            if name in arguments:
                kwargs[name] = arguments[name]
        if wants_context:
            kwargs[_CONTEXT_PARAM] = context
        return func(**kwargs)

    return executor


@dataclass
class ToolRegistrar:
    """Collects ``@register``-decorated functions into ``ToolSpec`` instances.

    One ``ToolRegistrar`` per module keeps the registered spec list local —
    ``build_builtin_registry`` reads ``.specs()`` to seed the ``ToolRegistry``.
    """

    _specs: list[ToolSpec] = field(default_factory=list)

    def register(
        self,
        *,
        capability: ToolCapability | None = None,
        permission_policy: PermissionPolicy | None = None,
        name: str | None = None,
        description: str | None = None,
    ) -> Callable[[Callable[..., ToolResult]], Callable[..., ToolResult]]:
        def decorator(func: Callable[..., ToolResult]) -> Callable[..., ToolResult]:
            schema, param_names, wants_context = _build_schema(func)
            resolved_name = name or func.__name__
            resolved_capability = capability or ToolCapability()
            resolved_policy = permission_policy or PermissionPolicy(kind=resolved_name)
            doc = inspect.cleandoc(func.__doc__ or "")
            resolved_description = enrich_tool_description(
                name=resolved_name,
                base_description=description or doc,
                capability=resolved_capability,
                permission_policy=resolved_policy,
            )
            spec = ToolSpec(
                name=resolved_name,
                description=resolved_description,
                input_schema=enrich_input_schema(resolved_name, schema),
                capability=resolved_capability,
                permission_policy=resolved_policy,
                validator=_passthrough_validator,
                executor=_make_executor(func, param_names, wants_context),
            )
            self._specs.append(spec)
            return func

        return decorator

    def specs(self) -> list[ToolSpec]:
        return list(self._specs)
