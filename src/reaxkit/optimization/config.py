"""Strict YAML validation and reproducible configuration resolution."""

from __future__ import annotations

import copy
import math
from pathlib import Path

import yaml

from .models import OptimizationError
from .registry import OPTIMIZERS, specification


class UniqueLoader(yaml.SafeLoader):
    """Reject duplicate YAML keys rather than silently taking the last value."""


def _mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise OptimizationError("YAML mapping keys must be strings")
        if key in result:
            raise OptimizationError(f"Duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def mapping(value, allowed, location):
    if not isinstance(value, dict):
        raise OptimizationError(f"{location}: expected a mapping")
    unknown = set(value) - set(allowed)
    if unknown:
        raise OptimizationError(f"{location}: unknown keys {sorted(unknown)}")
    return value


def check_value(value, rule, location):
    if value is None:
        return
    kind = rule["type"]
    valid = {
        "integer": type(value) is int,
        "number": type(value) in (int, float),
        "boolean": type(value) is bool,
        "string": isinstance(value, str) and bool(value.strip()),
        "path": isinstance(value, str) and bool(value.strip()),
        "list": isinstance(value, list) and bool(value) and all(isinstance(item, str) and item for item in value),
    }[kind]
    if not valid:
        raise OptimizationError(f"{location}: expected {kind}")
    if kind in ("integer", "number"):
        if not math.isfinite(value):
            raise OptimizationError(f"{location}: must be finite")
        for bound, compare in (("minimum", lambda left, right: left < right), ("maximum", lambda left, right: left > right)):
            if bound in rule and compare(value, rule[bound]):
                raise OptimizationError(f"{location}: violates {bound}={rule[bound]}")
    if "choices" in rule and value not in rule["choices"]:
        raise OptimizationError(f"{location}: choose from {rule['choices']}")


def path_value(value, base, location, exists=True, resolve_symlinks=True):
    check_value(value, {"type": "path"}, location)
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    path = path.resolve() if resolve_symlinks else path.absolute()
    if exists and not path.exists():
        raise OptimizationError(f"{location}: not found: {path}")
    return str(path)


def validate_conversion(value):
    mapping(value, ("progress_file", "columns", "skip_rows", "step_kind"), "conversion")
    if "progress_file" in value:
        check_value(value["progress_file"], {"type": "path"}, "conversion.progress_file")
        if value["progress_file"] is None:
            raise OptimizationError("conversion.progress_file: cannot be null")
        path = Path(value["progress_file"])
        if path.is_absolute() or ".." in path.parts:
            raise OptimizationError("conversion.progress_file: use a path inside the native output directory")
    columns = value.get("columns", {})
    mapping(columns, ("step", "score", "best_score", "elapsed_seconds"), "conversion.columns")
    for key, column in columns.items():
        check_value(column, {"type": "integer", "minimum": 0}, f"conversion.columns.{key}")
        if column is None:
            raise OptimizationError(f"conversion.columns.{key}: expected zero-based column index")
    if columns and not {"step", "score"} <= set(columns):
        raise OptimizationError("conversion.columns: step and score are required")
    if "skip_rows" in value:
        check_value(value["skip_rows"], {"type": "integer", "minimum": 0}, "conversion.skip_rows")
        if value["skip_rows"] is None:
            raise OptimizationError("conversion.skip_rows: cannot be null")
    if "step_kind" in value:
        check_value(value["step_kind"], {"type": "string", "choices": [
            "iteration", "generation", "parameter_row", "epoch", "evaluation",
        ]}, "conversion.step_kind")
    return value


def load_config(filename, overrides=None):
    source = Path(filename).resolve()
    try:
        original = source.read_text(encoding="utf-8")
        config = yaml.load(original, Loader=UniqueLoader)
    except (yaml.YAMLError, RecursionError) as error:
        raise OptimizationError(f"Invalid YAML in {source}: {error}") from error
    mapping(config, (
        "schema_version", "replicas", "optimizers", "seed", "output", "job",
        "inputs", "scheduler", "optimizer_settings", "collection_interval_seconds",
    ), "settings")
    for key in ("schema_version", "replicas", "optimizers", "output", "inputs"):
        if key not in config:
            raise OptimizationError(f"settings.{key}: required")
    if type(config["schema_version"]) is not int or config["schema_version"] != 1:
        raise OptimizationError("schema_version: expected 1")
    check_value(config["replicas"], {"type": "integer", "minimum": 1}, "replicas")
    if config["replicas"] is None:
        raise OptimizationError("replicas: required")
    selected = config["optimizers"]
    check_value(selected, {"type": "list"}, "optimizers")
    if selected is None or len(set(selected)) != len(selected):
        raise OptimizationError("optimizers: select unique optimizer IDs")
    for optimizer in selected:
        specification(optimizer)
    config.setdefault("seed", 42)
    check_value(config["seed"], {"type": "integer", "minimum": 0, "maximum": 2147483647}, "seed")
    config.setdefault("collection_interval_seconds", 30)
    check_value(config["collection_interval_seconds"], {"type": "number", "minimum": 1}, "collection_interval_seconds")
    if config["collection_interval_seconds"] is None:
        raise OptimizationError("collection_interval_seconds: cannot be null")
    base = source.parent
    config["output"] = path_value(config["output"], base, "output", exists=False)
    if config["output"] is None:
        raise OptimizationError("output: required")
    config["job"] = path_value(config.get("job"), base, "job")
    inputs = mapping(config["inputs"], ("ffield", "params", "trainset", "geo", "control", "params_bounds_order"), "inputs")
    for key in ("ffield", "params", "trainset", "geo", "control"):
        override = (overrides or {}).get(key)
        value = override if override is not None else inputs.get(key)
        inputs[key] = path_value(value, Path.cwd() if override is not None else base, f"inputs.{key}")
        if key != "control" and inputs[key] is None:
            raise OptimizationError(f"inputs.{key}: required")
        if inputs[key] is not None and not Path(inputs[key]).is_file():
            raise OptimizationError(f"inputs.{key}: expected a file")
    if inputs.get("params_bounds_order") not in ("min_max", "max_min"):
        raise OptimizationError("inputs.params_bounds_order: explicitly choose min_max or max_min")
    scheduler = mapping(config.setdefault("scheduler", {}), ("type", "profile"), "scheduler")
    scheduler.setdefault("type", "slurm")
    scheduler.setdefault("profile", "roar_collab")
    if scheduler != {"type": "slurm", "profile": "roar_collab"}:
        raise OptimizationError("scheduler: only slurm/roar_collab is supported")
    all_settings = mapping(config.setdefault("optimizer_settings", {}), OPTIMIZERS, "optimizer_settings")
    for optimizer, section in all_settings.items():
        prefix = f"optimizer_settings.{optimizer}"
        mapping(section, ("runtime", "job", "settings", "conversion", "prepared_inputs", "explicit_settings"), prefix)
        rules = specification(optimizer)["settings"]
        settings = mapping(section.setdefault("settings", {}), rules, prefix + ".settings")
        for key, value in settings.items():
            check_value(value, rules[key], prefix + ".settings." + key)
        runtime = mapping(section.setdefault("runtime", {}), ("mode", "command", "version", "engine_version", "environment", "source", "bridge_python", "timeout_seconds"), prefix + ".runtime")
        runtime.setdefault("mode", "external_runner")
        if runtime["mode"] not in ("native", "external_runner"):
            raise OptimizationError(prefix + ".runtime.mode: choose native or external_runner")
        command = runtime.get("command")
        if command is not None:
            check_value(command, {"type": "list"}, prefix + ".runtime.command")
            if runtime["mode"] == "external_runner" and command.count("{run_config}") != 1:
                raise OptimizationError(prefix + ".runtime.command: requires exactly one literal {run_config} argument")
            if runtime["mode"] == "native" and any("{run_config}" in argument for argument in command):
                raise OptimizationError(prefix + ".runtime.command: native entrypoints do not accept {run_config}")
            if command[0] == "{run_config}":
                raise OptimizationError(prefix + ".runtime.command: first argument must be the external runner executable")
            if any("{run_config}" in argument and argument != "{run_config}" for argument in command):
                raise OptimizationError(prefix + ".runtime.command: {run_config} must be a separate argument")
        environment = runtime.setdefault("environment", {})
        if not isinstance(environment, dict) or any(
            not isinstance(key, str) or not key or "=" in key or "\0" in key
            or not isinstance(value, str) or "\0" in value
            or key.startswith(("SLURM_", "SBATCH_"))
            for key, value in environment.items()
        ):
            raise OptimizationError(prefix + ".runtime.environment: expected string values without scheduler overrides")
        for key in ("version", "engine_version"):
            check_value(runtime.get(key), {"type": "string"}, prefix + ".runtime." + key)
        check_value(runtime.get("bridge_python"), {"type": "path"}, prefix + ".runtime.bridge_python")
        check_value(runtime.get("timeout_seconds"), {"type": "number", "minimum": 0.001}, prefix + ".runtime.timeout_seconds")
        validate_conversion(section.setdefault("conversion", {}))
        if optimizer not in selected:
            continue
        if command is None:
            raise OptimizationError(prefix + ".runtime.command: configure the native entrypoint or external runner argv")
        for argument in command:
            if "\0" in argument or "\n" in argument or "\r" in argument:
                raise OptimizationError(prefix + ".runtime.command: control characters are not allowed")
        runtime["source"] = path_value(runtime.get("source"), base, prefix + ".runtime.source")
        if runtime["source"] and not Path(runtime["source"]).is_dir():
            raise OptimizationError(prefix + ".runtime.source: expected a directory")
        if runtime.get("bridge_python"):
            runtime["bridge_python"] = path_value(runtime["bridge_python"], base, prefix + ".runtime.bridge_python", resolve_symlinks=False)
        section["job"] = path_value(section.get("job"), base, prefix + ".job") or config["job"]
        if not section["job"]:
            raise OptimizationError(prefix + ".job: a Slurm job file is required")
        from .schedulers.slurm import validate_job
        validate_job(Path(section["job"]))
        section["prepared_inputs"] = path_value(section.get("prepared_inputs"), base, prefix + ".prepared_inputs")
        if section["prepared_inputs"] and not Path(section["prepared_inputs"]).is_dir():
            raise OptimizationError(prefix + ".prepared_inputs: expected a directory")
        supplied = section.get("explicit_settings", sorted(settings))
        if not isinstance(supplied, list) or any(not isinstance(key, str) or key not in settings for key in supplied):
            raise OptimizationError(prefix + ".explicit_settings: expected configured setting names")
        for key, rule in rules.items():
            settings.setdefault(key, copy.deepcopy(rule["default"]))
            if rule["type"] == "path":
                settings[key] = path_value(settings[key], base, prefix + ".settings." + key, resolve_symlinks=key != "engine_python")
        section["explicit_settings"] = sorted(supplied)
        if optimizer == "jax_reaxff" and settings["use_valid"] and not all(settings[key] for key in ("valid_file", "valid_geo_file")):
            raise OptimizationError(prefix + ": validation requires valid_file and valid_geo_file")
        if optimizer == "indeedopt" and settings["short_samples"] and settings["engine"] != "jax-md":
            raise OptimizationError(prefix + ": short_samples requires the existing jax-md mode")
        if optimizer == "pibo":
            if settings["replicates"] != 1 or not settings["optimizers"] or len(settings["optimizers"]) != 1:
                raise OptimizationError(prefix + ": exactly one internal replicate and optimizer per orchestrated run")
            ucb_limit = settings["stage_ucb_until"] if settings["stage_ucb_until"] is not None else rules["stage_ucb_until"]["default"]
            ei_limit = settings["stage_ei_until"] if settings["stage_ei_until"] is not None else rules["stage_ei_until"]["default"]
            if ucb_limit > ei_limit:
                raise OptimizationError(prefix + ": stage_ucb_until must not exceed stage_ei_until")
        from .launching import validate_native
        validate_native(config, optimizer, section)
    for optimizer in selected:
        if optimizer not in all_settings:
            raise OptimizationError(f"optimizer_settings.{optimizer}: required")
    config["_source"] = str(source)
    config["_original"] = original
    return config


def generate_template(path):
    path = Path(path)
    document = {
        "schema_version": 1, "replicas": 1, "optimizers": ["jax_soppi"], "seed": 42,
        "output": "./optimization_runs", "job": "./roar_job.sh",
        "inputs": {
            "ffield": "./inputs/ffield", "params": "./inputs/params",
            "trainset": "./inputs/trainset.in", "geo": "./inputs/geo",
            "control": None, "params_bounds_order": "max_min",
        },
        "scheduler": {"type": "slurm", "profile": "roar_collab"},
        "collection_interval_seconds": 30,
    }
    text = (
        "# ReaxKit optimization schema v1. Fill paths before validation/submission.\n"
        "# job is your batch script path; it ends with: exec bash \"$1\"\n"
        "# No site account, modules, resource values, or engine paths are assumed.\n"
        "# Native mode appends verified optimizer flags to runtime.command; no optimizer libraries are imported.\n"
        "# External runners must consume {run_config}; INDEEDopt/EZFF and PIBO MoS2 require one.\n"
        "# Global replicas are independent jobs; native populations/trials are separate.\n"
        "# Null scientific settings preserve native/control-file defaults.\n"
    ) + yaml.safe_dump(document, sort_keys=False) + "optimizer_settings:\n"
    for optimizer in OPTIMIZERS:
        spec = specification(optimizer)
        text += f"  # {optimizer}: {spec['engine']}; seed support: {spec['seed']}.\n"
        commands = {
            "ams_mcff": ["${AMSBIN}/reaxff"], "ams_cmaes": ["${AMSBIN}/reaxff"],
            "standalone_soppi": ["/path/to/legacy/ffopt"],
            "jax_soppi": ["/path/to/soppi-env/bin/python", "-u", "-m", "soppi.cli"],
            "jax_reaxff": ["/path/to/jax-env/bin/python", "-u", "-m", "jaxreaxff.driver"],
            "ogolem": ["/path/to/java", "--add-modules=jdk.incubator.vector", "--enable-preview", "-jar", "/path/to/ogolem.jar"],
        }
        section = {
            "runtime": {"mode": "native" if optimizer in commands else "external_runner",
                        "command": commands.get(optimizer, ["/path/to/external/runner", "{run_config}"]),
                        "version": None, "engine_version": None, "source": None,
                        "bridge_python": None, "timeout_seconds": None, "environment": {}},
            "job": None, "prepared_inputs": None, "conversion": {},
        }
        text += "\n".join("  " + line for line in yaml.safe_dump({optimizer: section}, sort_keys=False).splitlines()) + "\n"
        text += "    settings:\n"
        for key, rule in spec["settings"].items():
            text += f"      # {rule['native']}; {rule['type']}.\n"
            if rule.get("support") == "external_runner_only":
                text += "      # Native script constant: requires configuration support in your external runner.\n"
            text += "\n".join("      " + line for line in yaml.safe_dump({key: rule["default"]}, sort_keys=False).splitlines()) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
    return path
