"""Registry for top-level workflow commands."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorkflowSpec:
    """Metadata for a top-level workflow command."""

    name: str
    module_path: str
    dispatch_mode: str = "task_subcommands"


WORKFLOW_REGISTRY: dict[str, WorkflowSpec] = {}


def register_workflow(
    name: str,
    *,
    module_path: str,
    dispatch_mode: str = "task_subcommands",
) -> WorkflowSpec:
    """Register a top-level workflow command."""
    spec = WorkflowSpec(name=name, module_path=module_path, dispatch_mode=dispatch_mode)
    WORKFLOW_REGISTRY[name] = spec
    return spec


def get_registered_workflows() -> dict[str, WorkflowSpec]:
    """Return all registered workflow commands."""
    return dict(WORKFLOW_REGISTRY)


register_workflow("intspec", module_path="reaxkit.workflows.meta.introspection_workflow", dispatch_mode="intspec_runner")
register_workflow("gui", module_path="reaxkit.workflows.meta.gui_workflow", dispatch_mode="kind_runner")
register_workflow("help", module_path="reaxkit.workflows.meta.help_workflow", dispatch_mode="kind_runner")
register_workflow("timeseries", module_path="reaxkit.workflows.timeseries_workflow", dispatch_mode="kind_runner")
