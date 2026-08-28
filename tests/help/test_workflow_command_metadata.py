from __future__ import annotations

from importlib import resources as ir

import yaml

from reaxkit.help.help_index_loader import build_help_relationship_report
from reaxkit.workflows.data.workflow_command_metadata import build_command_workflows


def _identity(name: str) -> str:
    return str(name).strip().lower().replace("-", "_")


def _load_workflow_map() -> dict:
    resource = ir.files("reaxkit.workflows.data").joinpath("workflow_dataclass_map.yaml")
    with resource.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def test_all_routed_commands_have_dedicated_workflow_metadata() -> None:
    document = _load_workflow_map()
    command_entries = {
        name: meta
        for name, meta in (document.get("workflows") or {}).items()
        if isinstance(meta, dict) and meta.get("parent_workflow")
    }

    covered = {
        _identity(value)
        for name, meta in command_entries.items()
        for value in [name, meta.get("command"), *(meta.get("aliases") or [])]
        if value
    }
    routed = {_identity(name) for name in (document.get("command_to_workflow_module") or {})}

    assert routed <= covered


def test_generated_command_workflows_are_current() -> None:
    document = _load_workflow_map()
    actual = {
        name: meta
        for name, meta in (document.get("workflows") or {}).items()
        if isinstance(meta, dict) and meta.get("parent_workflow")
    }

    assert actual == build_command_workflows(document)


def test_eos_help_surfaces_dedicated_workflow_command() -> None:
    report = build_help_relationship_report("eos", top_k=1)
    workflow_section = report.split("WORKFLOW LEVEL", 1)[1]

    assert "get_ffield_opt_eos" in workflow_section
    assert "reaxkit get_ffield_opt_eos -h" in workflow_section
