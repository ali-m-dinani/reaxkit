"""Reproducible registry and public-output inventory for migration reviews."""

import ast
from dataclasses import asdict, fields, is_dataclass
import importlib
import inspect
import json
from pathlib import Path
import typing
from functools import lru_cache

from reaxkit.core.runtime.execution_contracts import task_capabilities
from reaxkit.core.runtime.priority_task_manifest import PRIORITY_TASK_MANIFEST


@lru_cache(maxsize=None)
def _workflow_outputs(module_name):
    """Record dynamic public output expressions without executing workflows."""
    if not module_name:
        return []
    module_spec = importlib.util.find_spec(module_name)
    if module_spec is None or module_spec.origin is None:
        raise ValueError(f"Registered workflow module is unavailable: {module_name}")
    tree = ast.parse(Path(module_spec.origin).read_text(encoding="utf-8-sig"))
    outputs = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        method = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ""
        if method in {"to_csv", "to_parquet", "savefig", "write_text", "write_bytes", "write_extxyz", "write_workflow_csv", "workflow_csv_rows", "write_workflow_tables", "ArtifactSpec", "prepare_generator_output"}:
            outputs.append({"writer": method, "expression": ast.unparse(node), "line": node.lineno,
                            "tier": "declared" if method in {"ArtifactSpec", "write_workflow_tables"} else "core_compatibility"})
    return sorted(outputs, key=lambda record: record["line"])


def _table_inventory(result_class):
    names = []
    csv_tables = getattr(result_class, "csv_tables", None)
    if isinstance(csv_tables, property):
        import textwrap
        tree = ast.parse(textwrap.dedent(inspect.getsource(csv_tables.fget)))
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                names.extend(key.value for key in node.keys if isinstance(key, ast.Constant) and isinstance(key.value, str))
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant) and isinstance(target.slice.value, str):
                        names.append(target.slice.value)
    if not names and is_dataclass(result_class):
        hints = typing.get_type_hints(result_class)
        names = [field.name for field in fields(result_class) if "DataFrame" in str(hints.get(field.name))]
    tiers = getattr(result_class, "artifact_tiers", {})
    records = []
    for name in dict.fromkeys(names):
        tier = tiers.get(name, "core")
        format_name = "parquet" if tier in {"detail", "debug"} else "csv"
        records.append({"name": name, "filename": "result.csv" if names == ["table"] else f"{name}.{format_name}",
                        "legacy_filename": "result.csv" if names == ["table"] else f"{name}.csv",
                        "tier": tier, "format": format_name, "default_enabled": tier in {"core", "summary"}})
    return records


def _execution_note(cls, capabilities):
    """Explain intentional serial paths without treating missing work as safe."""
    if callable(getattr(cls, "run_blocks", None)):
        return "Global frame dependencies use task-specific disk-backed blocks; source input is read once."
    if callable(getattr(cls, "run_stream", None)):
        if capabilities.shape.value == "ordered_stateful_stream":
            return "Ordered scientific state is updated incrementally; independent frame workers would change history."
        if not capabilities.thread_safe:
            return "Bounded serial stream; the stateful reducer or external backend has no validated concurrent kernel."
        if not capabilities.automatic_parallel:
            return "Shared bounded map; automatic threading awaits a demonstrated benchmark benefit."
        return "Shared bounded map with validated automatic threading."
    data_type = getattr(getattr(cls, "required_data", None), "__name__", None)
    if data_type in {"SimulationData", "ElectricFieldData", "EregimeData", "PartialEnergyData", "RestraintData"}:
        return "Engine supplies a compact scalar-series table, not an atom-coordinate trajectory; retain its public table and use serial extraction."
    if data_type == "GeometryData":
        return "Isomer assignment compares the supplied geometry collection and shared equivalence classes globally."
    if data_type == "DataFrame":
        return "Dielectric autocorrelation and FFT require the complete compact dipole/time series; no generic frame splitting."
    if capabilities.shape.value == "single":
        return "One selected frame, file, report, or generator invocation; no independent trajectory workload."
    raise ValueError(f"Materialized task needs an audited execution reason: {cls.__module__}.{cls.__name__}")


def build_inventory():
    import reaxkit.analysis  # noqa: F401
    from reaxkit.core.registry.analysis_task_registry import TASK_REGISTRY
    from reaxkit.core.registry.command_catalog import get_registered_commands
    from reaxkit.core.registry.analysis_cli_routing_registry import get_registered_analysis_commands

    # Workflow-loaded analyses are part of the inventory even before their CLI
    # module is selected. Do not execute a scientific kernel to discover outputs.
    tasks = dict(TASK_REGISTRY)
    for entry in PRIORITY_TASK_MANIFEST:
        module = importlib.import_module(entry["module"])
        tasks[entry["command"]] = getattr(module, entry["class"])
    from reaxkit.core.runtime.analysis_task_manifest import ALL_GENERAL_TASKS
    declared = ALL_GENERAL_TASKS | {entry["command"] for entry in PRIORITY_TASK_MANIFEST}
    missing = set(tasks) - declared
    if missing:
        raise ValueError(f"Registered tasks need explicit capability declarations: {sorted(missing)}")
    analyses = []
    for name, cls in sorted(tasks.items()):
        module = importlib.import_module(cls.__module__)
        hints = typing.get_type_hints(inspect.unwrap(cls.run))
        result_cls = hints.get("return")
        if not inspect.isclass(result_cls):
            result_cls = getattr(module, cls.__name__.removesuffix("Task") + "Result", None)
        if result_cls is None:
            # Some task classes have historic names. Their request annotation
            # still connects them to an explicit result dataclass.
            for candidate in vars(module).values():
                if inspect.isclass(candidate) and is_dataclass(candidate) and candidate.__name__.endswith("Result"):
                    result_hints = typing.get_type_hints(candidate)
                    if hints.get("request") is not None and result_hints.get("request") == hints.get("request"):
                        result_cls = candidate
                        break
        outputs = _table_inventory(result_cls) if result_cls is not None else []
        capabilities = task_capabilities(cls())
        analyses.append({
            "task": name, "class": f"{cls.__module__}.{cls.__name__}",
            "capabilities": asdict(capabilities),
            "execution_note": _execution_note(cls, capabilities),
            "streaming": callable(getattr(cls, "run_stream", None)),
            "blocked": callable(getattr(cls, "run_blocks", None)),
            "required_data": getattr(getattr(cls, "required_data", None), "__name__", None),
            "result_class": getattr(result_cls, "__name__", None),
            "tables": outputs,
            "public_result_fields": [field.name for field in fields(result_cls)] if is_dataclass(result_cls) else [],
        })
    by_task = {record["task"]: record for record in analyses}
    routes = get_registered_analysis_commands()
    commands = []
    for name, spec in sorted(get_registered_commands().items()):
        route = routes.get(name)
        module = getattr(spec.target, "module_path", None) or getattr(route, "module_path", None)
        commands.append({"command": name, "kind": spec.kind, "aliases": list(spec.aliases),
                 "workflow": module,
                 "execution": by_task[name]["capabilities"] if name in by_task else
                     {"shape": "single" if spec.kind == "generator" else "global", "thread_safe": False},
                 "artifact_policy": "declared_task_tables" if name in by_task else "workflow_public_outputs",
                 "task_inventory": name if name in by_task else None,
                 "workflow_output_sites": _workflow_outputs(module),
                 "outputs": by_task[name]["tables"] if name in by_task else [],
                 "output_resolution": "task_result_tables" if name in by_task else "workflow_expressions_and_user_selected_paths"})
    return {"schema_version": 1, "analysis_tasks": analyses, "cli_commands": commands}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("docs/command_execution_inventory.json"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(build_inventory(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
