"""
Registry for top-level workflow commands.

**Usage context**

- Import these helpers from ReaxKit core modules when implementing CLI and workflow logic.
- Reuse the public APIs here to keep behavior consistent across commands and engines.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorkflowSpec:
    """
    Metadata for a top-level workflow command.
    
    
    Fields
    -----
    name : str
        Field value used by this structured record.
    module_path : str
        Field value used by this structured record.
    dispatch_mode : str, optional
        Field value used by this structured record.
    """

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
    """
    Register a top-level workflow command.
    
    This function is part of the ReaxKit core API and performs the operation
    described by its name and arguments.
    
    Parameters
    -----
    name : str
        Input parameter used by this function.
    module_path : str
        Input parameter used by this function.
    dispatch_mode : str, optional
        Input parameter used by this function.
    
    Returns
    -----
    WorkflowSpec
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.registry.workflow_cli_routing_registry import register_workflow
    # Configure required arguments for your case.
    result = register_workflow(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    spec = WorkflowSpec(name=name, module_path=module_path, dispatch_mode=dispatch_mode)
    WORKFLOW_REGISTRY[name] = spec
    return spec


def get_registered_workflows() -> dict[str, WorkflowSpec]:
    """
    Return all registered workflow commands.
    
    This function is part of the ReaxKit core API and performs the operation
    described by its name and arguments.
    
    Parameters
    -----
    None
    
    Returns
    -----
    dict[str, WorkflowSpec]
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.registry.workflow_cli_routing_registry import get_registered_workflows
    # Configure required arguments for your case.
    result = get_registered_workflows(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    return dict(WORKFLOW_REGISTRY)


register_workflow("intspec", module_path="reaxkit.workflows.meta.introspection_workflow", dispatch_mode="intspec_runner")
register_workflow("gui", module_path="reaxkit.workflows.meta.gui_workflow", dispatch_mode="kind_runner")
register_workflow("help", module_path="reaxkit.workflows.meta.help_workflow", dispatch_mode="kind_runner")
register_workflow("timeseries", module_path="reaxkit.workflows.timeseries_workflow", dispatch_mode="kind_runner")
