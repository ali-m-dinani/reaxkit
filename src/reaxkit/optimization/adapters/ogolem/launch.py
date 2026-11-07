"""Invoke the separately installed OGolem input bridge and Java optimizer."""

from pathlib import Path

from ...launching import native_plan, stage_common
from ...models import OptimizationError
from ...storage import read_json, write_json


def validate(config, section):
    runtime = section["runtime"]
    source = runtime.get("source")
    if not source or not (Path(source) / "tools" / "reaxff_bridge.py").is_file():
        raise OptimizationError("ogolem: runtime.source must contain tools/reaxff_bridge.py")
    if not runtime.get("bridge_python"):
        raise OptimizationError("ogolem: runtime.bridge_python must name its separately installed bridge interpreter")
    settings = section["settings"]
    if settings["backend"] == "custom" and not settings["evaluator_command"]:
        raise OptimizationError("ogolem custom backend requires evaluator_command (trusted evaluation-only argv)")
    if settings["backend"] != "custom" and settings["evaluator_command"]:
        raise OptimizationError("ogolem evaluator_command is only for backend: custom")
    if settings["ams_job_file"] and settings["backend"] != "ams":
        raise OptimizationError("ogolem ams_job_file requires backend: ams")
    if settings["timeout_seconds"] is None or settings["timeout_seconds"] <= 0:
        raise OptimizationError("ogolem timeout_seconds must be positive")
    if settings["backend"] == "ams" and not config["inputs"].get("control"):
        raise OptimizationError("ogolem AMS backend requires the user's compatible control input")


def prepare(request):
    native = Path(request["native_directory"])
    inputs = native / "bridge-inputs"
    bundle = native / "bundle"
    stage_common(request, inputs)
    runtime = request["runtime"]
    settings = request["settings"]
    bridge_python = runtime["bridge_python"]
    conversion = [bridge_python, str(Path(runtime["source"]) / "tools" / "reaxff_bridge.py"), "convert", str(inputs), str(bundle)]
    for key in ("profile", "backend", "iterations", "population", "engine_python", "ams_job_file"):
        if settings[key] is not None:
            conversion.extend(["--" + key.replace("_", "-"), str(settings[key])])
    for key, flag in (("include_initial", "--seed"), ("normalize_bounds", "--normalize-bounds")):
        if settings[key]:
            conversion.append(flag)
    command = list(runtime["command"]) + ["--adaptive", "fit.ogo", str(settings["workers"] or 1)]
    plan = native_plan(request, command, cwd=bundle, results_directory=str(bundle), environment={
        "REAXFF_BRIDGE_PYTHON": bridge_python, "OGO_EXTCALLCMD": str(bundle / "ogo_evaluator.sh"),
    }, warnings=[
        "OGolem --seed means initial-vector inclusion, not an RNG seed.",
        "The external bridge may generate its own example Slurm files; ReaxKit never submits those files.",
        "Choose workers and candidate engine resources together; do not multiply a full-allocation engine by concurrent workers.",
    ])
    plan["commands"].insert(0, {"argv": conversion, "cwd": str(native), "phase": "prepare"})
    plan["engine"] = {"ams": "SCM OldReaxFF", "jax-md": "JAX-MD ReaxFF", "custom": "User-supplied external evaluator"}[settings["backend"]]
    plan["objective"] = "custom_evaluator_fitness" if settings["backend"] == "custom" else "sum_squared_accuracy_divided_residuals"
    return plan


def after_prepare(request):
    bundle = Path(request["launch"]["results_directory"])
    path = bundle / "evaluator.json"
    settings = request["settings"]
    config = read_json(path)
    config["timeout_seconds"] = settings["timeout_seconds"]
    if settings["backend"] == "custom":
        config["command"] = settings["evaluator_command"]
    write_json(path, config)
