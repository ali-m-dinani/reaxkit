"""Backward-compatible imports for command alias resolution."""

from reaxkit.core.resolve.command_alias_resolver import (
    build_task_alias_index,
    is_known_command,
    is_known_task,
    registered_command_names,
    registered_task_names,
    resolve_command_name,
    resolve_task_name,
)

__all__ = [
    "build_task_alias_index",
    "is_known_command",
    "is_known_task",
    "registered_command_names",
    "registered_task_names",
    "resolve_command_name",
    "resolve_task_name",
]
