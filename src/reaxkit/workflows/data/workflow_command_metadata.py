"""Generate dedicated metadata for flat, user-invocable workflow commands.

The workflow map contains both workflow-family entries (for example,
``ffield_workflow``) and concrete commands (for example,
``get_ffield_opt_eos``).  This module derives the concrete entries from the
authoritative command routing map and enriches them with analyzer metadata
when an analysis task has the same command identity.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

import yaml

_START_MARKER = "  # BEGIN GENERATED COMMAND WORKFLOWS"
_END_MARKER = "  # END GENERATED COMMAND WORKFLOWS"


def _unique(values: Iterable[object]) -> list[str]:
    """Return non-empty strings in first-seen order."""
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _identity(name: str) -> str:
    """Normalize equivalent hyphen and underscore command spellings."""
    return str(name).strip().lower().replace("-", "_")


def _humanize(name: str) -> str:
    """Convert a command name into searchable prose."""
    return " ".join(part for part in _identity(name).split("_") if part)


def _command_from_ref(value: object) -> str:
    """Extract the command suffix from a ``module: command`` reference."""
    text = str(value or "").strip()
    return text.rsplit(":", 1)[-1].strip() if ":" in text else ""


def _analysis_task_for_group(
    names: list[str],
    tasks: dict[str, dict[str, Any]],
) -> tuple[str, dict[str, Any]] | tuple[None, None]:
    """Find analyzer metadata associated with one command spelling group."""
    by_identity = {_identity(key): (str(key), value) for key, value in tasks.items()}
    for name in names:
        match = by_identity.get(_identity(name))
        if match is not None:
            return match

    name_identities = {_identity(name) for name in names}
    for task_name, task_meta in tasks.items():
        related_command = _command_from_ref(task_meta.get("related_workflow_module"))
        if related_command and _identity(related_command) in name_identities:
            return str(task_name), task_meta
    return None, None


def _command_kinds() -> dict[str, str]:
    """Return command kind by normalized command identity."""
    from reaxkit.core.registry.analysis_cli_routing_registry import (
        get_registered_analysis_commands,
    )
    from reaxkit.core.registry.generator_cli_routing_registry import get_registered_generators
    from reaxkit.core.registry.workflow_cli_routing_registry import get_registered_workflows

    out: dict[str, str] = {}
    for name in get_registered_generators():
        out[_identity(name)] = "generator"
    for name in get_registered_analysis_commands():
        out[_identity(name)] = "analysis"
    for name in get_registered_workflows():
        out[_identity(name)] = "workflow"
    return out


def build_command_workflows(document: dict[str, Any]) -> OrderedDict[str, dict[str, Any]]:
    """Build canonical concrete-workflow entries from routed command names."""
    all_workflows = document.get("workflows") or {}
    family_workflows = {
        str(name): meta
        for name, meta in all_workflows.items()
        if isinstance(meta, dict) and not meta.get("parent_workflow")
    }
    routes = document.get("command_to_workflow_module") or {}

    analysis_path = (
        Path(__file__).resolve().parents[2]
        / "analysis"
        / "data"
        / "analysis_task_dataclass_map.yaml"
    )
    analysis_document = yaml.safe_load(analysis_path.read_text(encoding="utf-8")) or {}
    tasks = analysis_document.get("tasks") or {}
    kinds = _command_kinds()

    parent_by_module: dict[str, str] = {}
    parent_by_prefix: list[tuple[str, str]] = []
    for parent_name, parent_meta in family_workflows.items():
        implementation = str(parent_meta.get("implementation_module") or "")
        module_name = implementation.split(":", 1)[0].strip()
        if module_name:
            parent_by_module[module_name] = parent_name
        module_prefix = str(parent_meta.get("module_prefix") or "").strip().rstrip(".")
        if module_prefix:
            parent_by_prefix.append((module_prefix, parent_name))

    groups: OrderedDict[str, list[str]] = OrderedDict()
    for command_name in routes:
        groups.setdefault(_identity(command_name), []).append(str(command_name))

    generated: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for identity, names in groups.items():
        task_name, task_meta = _analysis_task_for_group(names, tasks)
        canonical = task_name if task_name in names else names[0]
        module_name = str(routes.get(canonical) or routes.get(names[0]) or "").strip()
        parent_name = parent_by_module.get(module_name)
        if parent_name is None:
            parent_name = next(
                (
                    candidate_parent
                    for prefix, candidate_parent in parent_by_prefix
                    if module_name == prefix or module_name.startswith(f"{prefix}.")
                ),
                module_name.rsplit(".", 1)[-1],
            )
        command_kind = kinds.get(identity, "workflow")

        task_meta = task_meta or {}
        human_name = _humanize(canonical)
        description = str(task_meta.get("description") or "").strip()
        if not description:
            description = f"Run the {human_name} workflow command provided by {parent_name}."

        aliases = _unique([
            *(name for name in names if name != canonical),
            *(task_meta.get("aliases") or []),
        ])
        tags = _unique([
            *(task_meta.get("tags") or []),
            human_name,
            _humanize(parent_name),
            f"{command_kind} workflow command",
        ])
        examples = _unique([
            *(task_meta.get("help_search_examples") or []),
            f'reaxkit help "{human_name}"',
            f"reaxkit {canonical} -h",
        ])

        entry: OrderedDict[str, Any] = OrderedDict()
        entry["parent_workflow"] = parent_name
        entry["implementation_module"] = f"{module_name}: {canonical}"
        entry["command"] = canonical
        entry["command_kind"] = command_kind
        if task_name:
            entry["analysis_task"] = task_name
        entry["description"] = description
        entry["tags"] = tags
        entry["notes"] = [f"How to use the workflow: reaxkit {canonical} -h"]
        entry["aliases"] = aliases
        entry["help_search_examples"] = examples
        generated[canonical] = entry

    return generated


def _yaml_block(entries: OrderedDict[str, dict[str, Any]]) -> str:
    """Serialize generated entries at the indentation of ``workflows:`` children."""
    plain_entries = {name: dict(meta) for name, meta in entries.items()}
    dumped = yaml.safe_dump(
        plain_entries,
        sort_keys=False,
        allow_unicode=True,
        width=100,
        default_flow_style=False,
    ).rstrip()
    indented = "\n".join(f"  {line}" if line else line for line in dumped.splitlines())
    return f"{_START_MARKER}\n{indented}\n{_END_MARKER}\n"


def update_workflow_map(path: Path | None = None) -> int:
    """Insert or replace the generated concrete-command block in the workflow map."""
    target = path or Path(__file__).with_name("workflow_dataclass_map.yaml")
    text = target.read_text(encoding="utf-8")
    document = yaml.safe_load(text) or {}
    entries = build_command_workflows(document)
    block = _yaml_block(entries)

    if _START_MARKER in text:
        before, remainder = text.split(_START_MARKER, 1)
        _, after = remainder.split(_END_MARKER, 1)
        updated = before + block + after.lstrip("\r\n")
    else:
        anchor = "\ndataclass_usage:\n"
        if anchor not in text:
            raise ValueError("Could not find dataclass_usage insertion point in workflow map.")
        updated = text.replace(anchor, f"\n{block}dataclass_usage:\n", 1)

    target.write_text(updated, encoding="utf-8", newline="\n")
    return len(entries)


def main() -> int:
    """Regenerate concrete workflow-command metadata in the packaged YAML map."""
    count = update_workflow_map()
    print(f"Updated {count} dedicated workflow command entries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
