"""Render per-run Slurm resources without changing the user's environment setup."""

import re
import shlex

from ..models import OptimizationError
from ..registry import JOB_ABBREVIATIONS


RESOURCE_FLAGS = {
    "nodes": ("--nodes", "-N"),
    "ntasks": ("--ntasks", "-n"),
    "ntasks_per_node": ("--ntasks-per-node",),
    "cpus_per_task": ("--cpus-per-task", "-c"),
    "time": ("--time", "-t"),
    "mem": ("--mem",),
}
SERIAL_RESOURCES = dict(nodes=1, ntasks=1, ntasks_per_node=1, cpus_per_task=1)


def validate_resources(resources):
    if not isinstance(resources, dict) or set(resources) - set(RESOURCE_FLAGS):
        raise OptimizationError("resources: expected nodes, ntasks, ntasks_per_node, cpus_per_task, time or mem")
    for key, value in resources.items():
        if key in SERIAL_RESOURCES:
            valid = type(value) is int and value >= 1
        elif key == "time":
            valid = isinstance(value, str) and re.fullmatch(r"(?:\d+-)?\d+:\d{2}:\d{2}", value)
            if valid:
                hours, minutes, seconds = value.split("-")[-1].split(":")
                valid = int(minutes) < 60 and int(seconds) < 60 and any(int(part) for part in re.split("[-:]", value))
        else:
            valid = isinstance(value, str) and re.fullmatch(r"[1-9]\d*(?:[KMGT])?", value, re.I)
        if not valid:
            raise OptimizationError(f"resources.{key}: invalid value {value!r}")
    if {"nodes", "ntasks", "ntasks_per_node"} <= resources.keys() and resources["ntasks"] > resources["nodes"] * resources["ntasks_per_node"]:
        raise OptimizationError("resources: ntasks exceeds nodes * ntasks_per_node")
    return resources


def resolve_resources(optimizer, shared, specific):
    validate_resources(shared)
    validate_resources(specific)
    result = dict(shared)
    if optimizer in ("standalone_soppi", "flocky"):
        result.update(SERIAL_RESOURCES)
        if any(key in specific and specific[key] != value for key, value in SERIAL_RESOURCES.items()):
            raise OptimizationError(f"{optimizer}: the integrated serial runner requires one node, task and CPU")
    result.update(specific)
    return validate_resources(result)


def render_job(text, optimizer, resources, job_name=None):
    validate_resources(resources)
    job_name = JOB_ABBREVIATIONS[optimizer] if job_name is None else job_name
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", job_name):
        raise OptimizationError("Invalid Slurm job name")
    overrides = {RESOURCE_FLAGS[key]: value for key, value in resources.items()}
    overrides[("--job-name", "-J")] = job_name
    header, body = [], []
    executable_seen = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            executable_seen = True
        if stripped.startswith("#SBATCH"):
            if executable_seen:
                raise OptimizationError("SBATCH directives must precede executable shell commands")
            tokens = shlex.split(stripped[7:], comments=True)
            if "mem" in resources and any(token.startswith(("--mem-per-cpu", "--mem-per-gpu")) for token in tokens):
                raise OptimizationError("Remove mem-per-cpu/mem-per-gpu before overriding total mem")
            remove = False
            for flags in overrides:
                for flag in flags:
                    matching = [token for token in tokens if token == flag or token.startswith(flag + "=")
                                or (len(flag) == 2 and token.startswith(flag) and len(token) > 2)]
                    if matching:
                        if len(tokens) > 2 or (len(tokens) == 2 and tokens[0] != flag):
                            raise OptimizationError("Use one Slurm option per SBATCH line when overriding resources")
                        remove = True
            if remove:
                continue
        (body if executable_seen else header).append(line)
    additions = [f"#SBATCH {flags[0]}={value}" for flags, value in overrides.items()]
    assignment = "export REAXKIT_OPTIMIZER=" + shlex.quote(optimizer)
    return "\n".join(header + additions + [assignment] + body) + "\n"
