"""Core orchestration components."""

from .analysis_executor import AnalysisExecutor
from .command_registry import (
    COMMAND_REGISTRY,
    CommandSpec,
    command,
    get_registered_commands,
    register_command,
    register_generator_command,
)
from .task_resolution_using_alias import (
    is_known_command,
    is_known_task,
    resolve_command_name,
    resolve_task_name,
)

__all__ = [
    "AnalysisExecutor",
    "COMMAND_REGISTRY",
    "CommandSpec",
    "command",
    "get_registered_commands",
    "is_known_command",
    "is_known_task",
    "register_command",
    "register_generator_command",
    "resolve_command_name",
    "resolve_task_name",
]
