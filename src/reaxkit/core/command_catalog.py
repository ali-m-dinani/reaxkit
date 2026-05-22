"""Catalog for top-level user-facing commands."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Iterable
import importlib.resources as ir

import yaml

from reaxkit.core.analysis_cli_routing_registry import get_registered_analysis_commands
from reaxkit.core.generator_cli_routing_registry import get_registered_generators
from reaxkit.core.analysis_task_registry import TASK_REGISTRY
from reaxkit.core.workflow_cli_routing_registry import get_registered_workflows


@dataclass(frozen=True)
class CommandSpec:
    """Metadata for a user-facing command."""

    name: str
    kind: str
    target: object | None = None
    aliases: tuple[str, ...] = field(default_factory=tuple)
    help_text: str = ""


COMMAND_REGISTRY: dict[str, CommandSpec] = {}


def register_command(
    name: str,
    *,
    kind: str,
    target: object | None = None,
    aliases: Iterable[str] = (),
    help_text: str = "",
) -> CommandSpec:
    """Register a user-facing command."""
    spec = CommandSpec(
        name=name,
        kind=kind,
        target=target,
        aliases=tuple(aliases),
        help_text=str(help_text or ""),
    )
    COMMAND_REGISTRY[name] = spec
    return spec


def command(name: str, *, kind: str, aliases: Iterable[str] = (), help_text: str = "") -> Callable:
    """Decorator that registers a class or function as a command target."""

    def wrapper(target):
        register_command(name, kind=kind, target=target, aliases=aliases, help_text=help_text)
        return target

    return wrapper


def register_generator_command(
    name: str,
    *,
    target: object | None = None,
    aliases: Iterable[str] = (),
    help_text: str = "",
) -> CommandSpec:
    """Convenience wrapper for generator-backed commands."""
    return register_command(name, kind="generator", target=target, aliases=aliases, help_text=help_text)


@lru_cache(maxsize=1)
def _load_packaged_command_metadata() -> dict[str, dict[str, object]]:
    """Load packaged command metadata from help YAML sources."""
    pkg = "reaxkit.help.data"
    rel = "help_search_index.yaml"

    try:
        with ir.files(pkg).joinpath(rel).open("r", encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
    except FileNotFoundError:
        # help_search_index.yaml was deprecated in favor of layer-specific maps.
        # Return an empty metadata map and let registries provide command names.
        return {}

    raw_commands = doc.get("commands") or {}
    out: dict[str, dict[str, object]] = {}
    for name, meta in raw_commands.items():
        meta_map = dict(meta or {})
        aliases = meta_map.get("aliases") or ()
        if not isinstance(aliases, (list, tuple)):
            aliases = (aliases,)
        out[str(name)] = {
            "kind": str(meta_map.get("kind") or "analysis"),
            "aliases": tuple(str(alias) for alias in aliases if str(alias)),
            "help_text": str(meta_map.get("desc") or meta_map.get("help") or ""),
        }
    return out


def _register_packaged_commands() -> None:
    """Populate the catalog from packaged command metadata."""
    for name, meta in _load_packaged_command_metadata().items():
        register_command(
            name,
            kind=str(meta["kind"]),
            aliases=tuple(meta["aliases"]),
            help_text=str(meta["help_text"]),
        )


def get_registered_commands(include_analysis_tasks: bool = True) -> dict[str, CommandSpec]:
    """Return all currently known commands."""
    commands = dict(COMMAND_REGISTRY)
    workflow_specs = get_registered_workflows()
    for name, spec in workflow_specs.items():
        meta = commands.get(name)
        commands[name] = CommandSpec(
            name=name,
            kind="workflow",
            target=spec,
            aliases=meta.aliases if meta is not None else tuple(),
            help_text=meta.help_text if meta is not None else "",
        )

    generator_specs = get_registered_generators()
    for name, spec in generator_specs.items():
        meta = commands.get(name)
        commands[name] = CommandSpec(
            name=name,
            kind="generator",
            target=spec,
            aliases=meta.aliases if meta is not None else tuple(),
            help_text=meta.help_text if meta is not None else "",
        )

    if include_analysis_tasks:
        command_metadata = _load_packaged_command_metadata()
        analysis_routes = get_registered_analysis_commands()
        for name, target in TASK_REGISTRY.items():
            route = analysis_routes.get(name)
            if name in commands:
                spec = commands[name]
                if spec.target is None:
                    commands[name] = CommandSpec(
                        name=spec.name,
                        kind=spec.kind,
                        target=target if route is None else {"task": target, "route": route},
                        aliases=spec.aliases,
                        help_text=spec.help_text,
                    )
                continue
            meta = command_metadata.get(name) or {}
            commands[name] = CommandSpec(
                name=name,
                kind=str(meta.get("kind") or "analysis"),
                target=target if route is None else {"task": target, "route": route},
                aliases=tuple(meta.get("aliases") or ()),
                help_text=str(meta.get("help_text") or ""),
            )
    return commands


_register_packaged_commands()
