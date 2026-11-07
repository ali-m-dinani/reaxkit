"""Map jaxreaxff.driver arguments and evaluation-only reports."""

from pathlib import Path

from ...launching import add_settings, native_plan, stage_common, validate_jax_data
from ...models import OptimizationError
from ...registry import specification


def validate(config, section):
    settings = section["settings"]
    validate_jax_data(config["inputs"]["trainset"], config["inputs"]["geo"])
    if settings["use_valid"]:
        validate_jax_data(settings["valid_file"], settings["valid_geo_file"])
    if settings["machine_report"] and settings["num_trials"] != 0:
        raise OptimizationError("jax_reaxff.machine_report requires num_trials: 0 (native evaluation-only mode)")
    if config["inputs"].get("control"):
        raise OptimizationError("jax_reaxff does not read control; omit it and configure the native minimization/engine settings explicitly")


def prepare(request):
    inputs = stage_common(request, Path(request["native_directory"]) / "inputs")
    command = list(request["runtime"]["command"])
    for key, flag in (("ffield", "--init_FF"), ("params", "--params"), ("trainset", "--train_file"), ("geo", "--geo")):
        command.extend([flag, inputs[key]])
    command.extend(["--out_folder", request["native_directory"]])
    add_settings(command, request["settings"], specification("jax_reaxff")["settings"], excluded=("machine_report",))
    if request["seed"] is not None:
        command.extend(["--seed", str(request["seed"])])
    if request["settings"]["machine_report"]:
        command.extend(["--machine_report", str(Path(request["native_directory"]) / "machine_report.json")])
    return native_plan(request, command, seed_forwarded=request["seed"], warnings=[
        "Inputs must already use JAX-ReaxFF-supported training and geometry syntax; no dataset filtering/conversion is performed."
    ])
