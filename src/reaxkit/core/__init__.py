"""Core orchestration components."""

from .analysis_executor import AnalysisExecutor
from .analysis_cli_routing_registry import get_registered_analysis_commands, register_analysis_command
from .analysis_task_registry import TASK_REGISTRY, register_task
from .command_catalog import (
    COMMAND_REGISTRY,
    CommandSpec,
    command,
    get_registered_commands,
    register_command,
    register_generator_command,
)
from .command_alias_resolver import (
    is_known_command,
    is_known_task,
    normalize_command_token,
    resolve_command_name,
    resolve_task_name,
)
from .generator_cli_routing_registry import get_registered_generators, register_generator
from .workflow_cli_routing_registry import get_registered_workflows, register_workflow

__all__ = [
    "AnalysisExecutor",
    "get_registered_analysis_commands",
    "TASK_REGISTRY",
    "COMMAND_REGISTRY",
    "CommandSpec",
    "command",
    "get_registered_commands",
    "get_registered_generators",
    "get_registered_workflows",
    "is_known_command",
    "is_known_task",
    "normalize_command_token",
    "register_command",
    "register_analysis_command",
    "register_task",
    "register_generator",
    "register_generator_command",
    "register_workflow",
    "resolve_command_name",
    "resolve_task_name",
]
