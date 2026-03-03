"""Registry for top-level user-facing commands."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

from reaxkit.core.task_registry import TASK_REGISTRY


@dataclass(frozen=True)
class CommandSpec:
    """Metadata for a user-facing command."""

    name: str
    kind: str
    target: object | None = None
    aliases: tuple[str, ...] = field(default_factory=tuple)


COMMAND_REGISTRY: dict[str, CommandSpec] = {}


def register_command(
    name: str,
    *,
    kind: str,
    target: object | None = None,
    aliases: Iterable[str] = (),
) -> CommandSpec:
    """Register a user-facing command."""
    spec = CommandSpec(
        name=name,
        kind=kind,
        target=target,
        aliases=tuple(aliases),
    )
    COMMAND_REGISTRY[name] = spec
    return spec


def command(name: str, *, kind: str, aliases: Iterable[str] = ()) -> Callable:
    """Decorator that registers a class or function as a command target."""

    def wrapper(target):
        register_command(name, kind=kind, target=target, aliases=aliases)
        return target

    return wrapper


def register_generator_command(name: str, *, target: object | None = None, aliases: Iterable[str] = ()) -> CommandSpec:
    """Convenience wrapper for generator-backed commands."""
    return register_command(name, kind="generator", target=target, aliases=aliases)


def get_registered_commands(include_analysis_tasks: bool = True) -> dict[str, CommandSpec]:
    """Return all currently known commands."""
    commands = dict(COMMAND_REGISTRY)
    if include_analysis_tasks:
        for name, target in TASK_REGISTRY.items():
            commands.setdefault(name, CommandSpec(name=name, kind="analysis", target=target))
    return commands


# Example command declarations for the top-level CLI namespace.
register_command(
    "msd",
    kind="analysis",
    aliases=("mean-square-displacement", "mean_square_displacement"),
)
register_command(
    "rdf",
    kind="analysis",
    aliases=("radial-distribution-function", "radial_distribution_function"),
)
register_command(
    "rdf_property",
    kind="analysis",
    aliases=("rdf-property", "rdfproperty"),
)
