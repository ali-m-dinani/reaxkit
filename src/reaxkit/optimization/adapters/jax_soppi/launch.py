"""Map the inspected soppi.cli interface without importing JAX."""

from pathlib import Path

from ...launching import add_settings, native_plan, stage_common, validate_jax_data
from ...models import OptimizationError
from ...registry import specification


def validate(config, section):
    validate_jax_data(config["inputs"]["trainset"], config["inputs"]["geo"])
    for key in ("jax_reaxff_root", "jax_md_root"):
        if not section["settings"].get(key) or not Path(section["settings"][key]).is_dir():
            raise OptimizationError(f"jax_soppi: explicitly configure the {key} source directory")


def prepare(request):
    inputs = stage_common(request, Path(request["native_directory"]) / "inputs")
    command = list(request["runtime"]["command"])
    for key, filename in inputs.items():
        command.extend(["--" + key, filename])
    command.extend(["--output", request["native_directory"], "--bounds-order", "min_max"])
    add_settings(command, request["settings"], specification("jax_soppi")["settings"])
    return native_plan(request, command, warnings=["Deterministic search has no native RNG seed flag; replicas may be identical."])
